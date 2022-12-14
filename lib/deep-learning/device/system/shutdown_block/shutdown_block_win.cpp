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
#include "deep-learning/device/system/shutdown_block/shutdown_block_win.hpp"

// Deep learning:
#include "deep-learning/io/logger.hpp"

// Standard:
#include <stdexcept>

namespace DL::Sys {
ShutdownBlock::ShutdownBlock(std::wstring const &wnd_name,
                             std::wstring const &cls_name)
    : _wnd_name(wnd_name), _cls_name(cls_name) {
  if (shutdownblock != nullptr)
    throw std::runtime_error(
        "ShutdownBlock has already been initiated somewhere else.");
  shutdownblock = this;
}

BOOL WINAPI ShutdownBlock::WINAPI__ConsoleCtrlHandler(DWORD dwCtrlType) {
  if (dwCtrlType == CTRL_C_EVENT || dwCtrlType == CTRL_BREAK_EVENT ||
      dwCtrlType == CTRL_CLOSE_EVENT) {
    shutdownblock->console_ctrl_handler(dwCtrlType);
    return TRUE;
  }

  return FALSE;
}

void ShutdownBlock::console_ctrl_handler(DWORD const dwCtrlType) {
  if (this->_initialized == false) return;

  if (dwCtrlType == CTRL_C_EVENT || dwCtrlType == CTRL_BREAK_EVENT ||
      dwCtrlType == CTRL_CLOSE_EVENT) {
    SendMessageW(this->_hwnd, WM_CLOSE, 0, 0);
  }
}

LRESULT CALLBACK CallBack_WndProc(HWND hwnd, UINT msg, WPARAM wparam,
                                  LPARAM lparam) {
  PAINTSTRUCT paint_struct;

  HDC hdc;

  switch (msg) {
      // case WM_CLOSE: return(FALSE); // 16
      // case WM_QUIT: return(FALSE); // 18
      // case WM_ENDSESSION: return(FALSE); // 22
      // case WM_TIMECHANGE: return(FALSE); // 30
      // case SPI_SETDOUBLECLKHEIGHT: return(FALSE); // 30
      // case VK_ACCEPT: return(FALSE); // 30
      // case WM_GETMINMAXINFO: return(FALSE); // 36
      // case SPI_SETSTICKYKEYS: return(FALSE); // 59
      // case WM_GETICON: return(FALSE); // 127
      // case WM_NCCREATE: return(FALSE); // 129
      // case WM_NCDESTROY: return(FALSE); // 130
      // case SPI_GETWINARRANGING: return(FALSE); // 130
      // case VK_F19: return(FALSE); // 130
      // case CF_DSPBITMAP: return(FALSE); // 130
      // case WM_NCCALCSIZE: return(FALSE); // 131
      // case WM_NCACTIVATE: return(FALSE); // 134
      // case SPI_GETDOCKMOVING: return(FALSE); // 144
      // case VK_NUMLOCK: return(FALSE); // 144
      // case WM_SYSTIMER: return(FALSE); // 280
      // case WM_DEVICECHANGE: return(FALSE); // 537
      // case WM_DWMNCRENDERINGCHANGED: return(FALSE); // 799
      // case WM_DWMCOLORIZATIONCOLORCHANGED: return(FALSE); // 800

    case WM_CREATE:  // 1
      ShutdownBlockReasonCreate(hwnd, L"DL needs to close properly.");
      break;
    case WM_DESTROY:  // 2
      PostQuitMessage(0);
      break;
    case WM_PAINT:  // 15
      hdc = BeginPaint(hwnd, &paint_struct);
      EndPaint(hwnd, &paint_struct);
      break;
    case WM_QUERYENDSESSION:  // 17
      shutdownblock->query_shutdown();
      return FALSE;
    default:
      return DefWindowProc(hwnd, msg, wparam, lparam);
  }

  return 0;
}

ATOM ShutdownBlock::register_class(HINSTANCE handle) {
  WNDCLASSEX window;

  window.cbSize = sizeof(WNDCLASSEX);
  window.style = CS_HREDRAW | CS_VREDRAW;
  window.lpfnWndProc = CallBack_WndProc;
  window.cbClsExtra = 0;
  window.cbWndExtra = 0;
  window.hInstance = handle;
  window.hIcon = LoadIcon(NULL, IDI_APPLICATION);
  window.hCursor = LoadCursor(NULL, IDC_ARROW);
  window.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
  window.lpszMenuName = NULL;
  window.lpszClassName = this->_cls_name.c_str();
  window.hIconSm = NULL;

  return RegisterClassExW(&window);
}

BOOL ShutdownBlock::initiate_instance(HINSTANCE handle, int nCmdShow) {
  if (this->_wnd_name.empty()) {
    ERR(L"Window name is empty.");
    return FALSE;
  } else if (this->_cls_name.empty()) {
    ERR(L"Class name is empty.");
    return FALSE;
  }

  RECT rect = {0, 0, 512, 512};

  AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, TRUE);

  this->_hwnd = CreateWindow(this->_cls_name.c_str(), this->_wnd_name.c_str(),
                             WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
                             rect.right - rect.left, rect.bottom - rect.top,
                             NULL, NULL, handle, NULL);

  if (this->_hwnd == NULL) {
    ERR(L"An error has been triggered from the `CreateWindow()` function.");
    return FALSE;
  }

  ShowWindow(this->_hwnd, nCmdShow);

  BOOL ret_code;
  if ((ret_code = UpdateWindow(this->_hwnd)) == FALSE) {
    ERR(L"An error has been triggered from the "
        L"`UpdateWindow() -> %d` function.",
        ret_code);
    return FALSE;
  }

  return TRUE;
}

void ShutdownBlock::query_shutdown(void) {
  this->_preparing_for_shutdown.store(true);
}

bool ShutdownBlock::preparing_for_shutdown(void) const {
  return this->_preparing_for_shutdown.load();
}

bool ShutdownBlock::block(bool const use_ctrl_handler) {
  if (this->_initialized || this->preparing_for_shutdown()) return false;

  BOOL ret_code;

  if (use_ctrl_handler && (ret_code = SetConsoleCtrlHandler(
                               static_cast<PHANDLER_ROUTINE>(
                                   ShutdownBlock::WINAPI__ConsoleCtrlHandler),
                               TRUE)) == FALSE) {
    ERR(L"An error has been triggered from the "
        "`SetConsoleCtrlHandler() -> %d` function.",
        ret_code);
    return false;
  }

  HINSTANCE hinstance(GetModuleHandleW(NULL));

  if (hinstance == NULL) {
    ERR(L"An error has been triggered from the `GetModuleHandle()` function.");
    return false;
  }

  this->register_class(hinstance);

  if ((ret_code = this->initiate_instance(hinstance, SW_HIDE)) == FALSE) {
    ERR(L"An error has been triggered from the "
        L"`initiate_instance() -> %d` function.",
        ret_code);
    return false;
  }

  this->_hinstance = hinstance;
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

  BOOL ret_code;
  if ((ret_code = ShutdownBlockReasonDestroy(this->_hwnd)) == FALSE) {
    ERR(L"An error has been triggered from the "
        L"`ShutdownBlockReasonDestroy() -> %d` function.",
        ret_code);
    return false;
  }

  this->_preparing_for_shutdown.store(false);
  this->_initialized = false;

  return true;
}

bool ShutdownBlock::peek_message(void) {
  if (this->preparing_for_shutdown()) return true;

  BOOL ret_code;

  MSG msg;

  while ((ret_code = PeekMessageW(&msg, this->_hwnd, 0, 0, PM_REMOVE)) !=
         FALSE) {
    if (ret_code == FAIL) {
      ERR(L"An error has been triggered from the "
          L"`PeekMessage() -> %d` function.",
          ret_code);
      return false;
    }

    TranslateMessage(&msg);
    DispatchMessageW(&msg);
  }

  if (msg.message == WM_QUIT) this->query_shutdown();

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

ShutdownBlock::~ShutdownBlock(void) { this->unblock(); }
}  // namespace DL::Sys