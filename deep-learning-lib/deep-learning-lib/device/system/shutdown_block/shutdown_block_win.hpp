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
#include <windows.h>

#include <atomic>
#include <string>
#include <thread>

namespace DL::Sys {
class ShutdownBlock {
 public:
  ShutdownBlock(std::wstring const &wnd_name, std::wstring const &cls_name);
  ~ShutdownBlock(void);

  void console_ctrl_handler(DWORD const dwCtrlType);
  void query_shutdown(void);

  bool preparing_for_shutdown(void) const;
  bool block(bool const use_ctrl_handler);
  bool unblock(void);

 private:
  bool _initialized = false;
  bool peek_message(void);

  std::atomic<bool> _preparing_for_shutdown = false;

  ATOM register_class(HINSTANCE handle);

  static BOOL WINAPI WINAPI__ConsoleCtrlHandler(DWORD dwCtrlType);

  BOOL initiate_instance(HINSTANCE handle, int nCmdShow);

  void run(void);

  std::wstring _wnd_name = L"";
  std::wstring _cls_name = L"";

  HWND _hwnd = NULL;

  HINSTANCE _hinstance = NULL;

  std::thread _thread;
};

inline ShutdownBlock *shutdownblock = nullptr;
}  // namespace DL::Sys