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

// Systemd:
#include <systemd/sd-bus.h>

// Standard:
#include <atomic>
#include <string>
#include <thread>

namespace DL::Sys {
class ShutdownBlock {
 public:
  ShutdownBlock(std::string const &wnd_name);
  ~ShutdownBlock(void);

  void query_shutdown(void);

  bool preparing_for_shutdown(void) const;
  bool block(bool const use_ctrl_handler);
  bool unblock(void);

 private:
  bool _initialized = false;
  bool peek_message(void);
  bool check_systemd_version(void) const;

  std::atomic<bool> _preparing_for_shutdown = false;

  void run(void);

  std::string _wnd_name = "";

  sd_bus *_sd_bus = NULL;

  sd_bus_message *_sd_bus_handle = NULL;
  sd_bus_message *_sd_bus_message = NULL;

  sd_bus_error _sd_bus_error = SD_BUS_ERROR_NULL;

  std::thread _thread;
};

inline ShutdownBlock *shutdownblock = nullptr;
}  // namespace DL::Term