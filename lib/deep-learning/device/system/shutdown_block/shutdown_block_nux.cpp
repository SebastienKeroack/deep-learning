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
#include "deep-learning/device/system/shutdown_block/shutdown_block_nux.hpp"

// Deep learning:
#include "deep-learning/io/logger.hpp"
#include "deep-learning/io/term/command.hpp"

// Standard:
#include <stdexcept>

using namespace DL::Term;

namespace DL::Sys {
ShutdownBlock::ShutdownBlock(std::string const &wnd_name)
    : _preparing_for_shutdown(false), _wnd_name(wnd_name) {
  if (shutdownblock != nullptr)
    throw std::runtime_error(
        "ShutdownBlock has already been initiated somewhere else.");
  shutdownblock = this;
}

bool ShutdownBlock::check_systemd_version(void) const {
  std::wstring output(check_output(L"systemd --version"));

  if (output.empty()) {
    ERR(L"An error has been triggered from the "
        L"`check_output(\"systemd --version\")` function.");
    return false;
  }

  std::wstring::size_type const char_pos(output.find_first_of(L" "));

  if (char_pos == std::wstring::npos) {
    ERR(L"An error has been triggered from the "
        L"`find_first_of(\" \")` function.");
    return false;
  }

  // Substring "systemd ".
  output = output.substr(char_pos + 1);

  int systemd_ver(0);

  try {
    systemd_ver = std::stoi(output);
  } catch (std::exception &e) {
    ERR(L"An error has been triggered from the "
        L"`std::stoi(%ls) -> %ls` function.",
        output.c_str(), e.what());
    return false;
  }

  if (systemd_ver < 220) {
    ERR(L"Systemd current version %d need to be update to the "
        L"version 220 or greater.",
        systemd_ver);
    return false;
  }

  return true;
}

void ShutdownBlock::query_shutdown(void) {
  this->_preparing_for_shutdown.store(true);
}

bool ShutdownBlock::preparing_for_shutdown(void) const {
  return this->_preparing_for_shutdown.load();
}

bool ShutdownBlock::block(bool const use_ctrl_handler) {
  if (this->check_systemd_version() == false) {
    ERR(L"An error has been triggered from the "
        L"`check_systemd_version()` function.");
    return false;
  }

  if (this->_initialized || this->preparing_for_shutdown()) return false;

  int ret_code;

  // Connect to the system bus.
  if (this->_sd_bus == NULL &&
      (ret_code = sd_bus_open_system(&this->_sd_bus)) < 0) {
    ERR(L"An error has been triggered from the "
        L"`sd_bus_open_system() -> %d` function.",
        ret_code);

    sd_bus_error_free(&this->_sd_bus_error);

    return false;
  }

  /* Issue the method call and store the handle:
  / [bash]
  / $ busctl call org.freedesktop.login1 /org/freedesktop/login1 \
  org.freedesktop.login1.Manager \
  Inhibit ssss shutdown MyAppName "MyReason" delay */
  if ((ret_code = sd_bus_call_method(
           this->_sd_bus,
           "org.freedesktop.login1",          // Service to contact.
           "/org/freedesktop/login1",         // Object path.
           "org.freedesktop.login1.Manager",  // Interface name.
           "Inhibit",                         // Method name.
           &this->_sd_bus_error,              // Object to return error in.
           &this->_sd_bus_handle,             // Return message on success.
           "ssss",                            // Input signature.
           "shutdown",                        // argument: what.
           this->_wnd_name.c_str(),           // argument: who.
           "DL needs to close properly.",     // argument: why.
           "delay")) < 0)                     // argument: mode.
  {
    ERR(L"An error has been triggered from the "
        L"`sd_bus_call_method() -> (%d, %ls)` function.",
        ret_code, this->_sd_bus_error.message);

    sd_bus_unref(this->_sd_bus);

    sd_bus_error_free(&this->_sd_bus_error);

    return false;
  }

  this->_thread = std::thread(&ShutdownBlock::run, this);
  this->_initialized = true;

  return true;
}

bool ShutdownBlock::unblock(void) {
  if (this->_initialized == false) return false;

  if (this->_thread.joinable()) {
    this->_preparing_for_shutdown.store(true);
    this->_thread.join();
  }

  sd_bus_message_unref(this->_sd_bus_handle);

  this->_preparing_for_shutdown.store(false);
  this->_initialized = false;

  return true;
}

bool ShutdownBlock::peek_message(void) {
  if (this->preparing_for_shutdown()) return true;

  int ret_code, ret_result;

  /* Issue the method call and store the value in a boolean:
  / [bash]
  / $ busctl get-property org.freedesktop.login1 /org/freedesktop/login1 \
  org.freedesktop.login1.Manager PreparingForSleep */
  if ((ret_code = sd_bus_get_property(
           this->_sd_bus,
           "org.freedesktop.login1",          // Service to contact.
           "/org/freedesktop/login1",         // Object path.
           "org.freedesktop.login1.Manager",  // Interface name.
           "PreparingForShutdown",            // Method name.
           &this->_sd_bus_error,              // Object to return error in.
           &this->_sd_bus_message,            // Reply.
           "b"                                // Input signature.
           )) < 0)                            // Output.
  {
    ERR(L"An error has been triggered from the "
        L"`sd_bus_get_property() -> (%d, %ls)` function.",
        ret_code, this->_sd_bus_error.message);

    sd_bus_error_free(&this->_sd_bus_error);

    return false;
  }

  // Read.
  if ((ret_code =
           sd_bus_message_read(this->_sd_bus_message, "b", &ret_result)) < 0) {
    ERR(L"An error has been triggered from the "
        L"`sd_bus_message_read() -> %d` function.",
        ret_code);
    return false;
  }

  if (ret_result != 0) this->query_shutdown();

  return true;
}

void ShutdownBlock::run(void) {
  do {
    if (this->peek_message() == false) {
      ERR(L"An error has been triggered from the `peek_message()` function.");
      return;
    }

    std::this_thread::sleep_for(std::chrono::seconds(3));
  } while (this->preparing_for_shutdown() == false);
}

ShutdownBlock::~ShutdownBlock(void) {
  this->unblock();

  if (this->_sd_bus != NULL) sd_bus_unref(this->_sd_bus);

  sd_bus_error_free(&this->_sd_bus_error);
}
}  // namespace DL::Sys