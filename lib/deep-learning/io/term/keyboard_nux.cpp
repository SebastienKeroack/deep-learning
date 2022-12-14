/* Copyright 2016, 2019 Sébastien Kéroack. All Rights Reserved.

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
#include "deep-learning/io/term/keyboard_nux.hpp"

// Standard:
#include <sys/ioctl.h>
#include <termios.h>

#include <iostream>

namespace DL::Term {
Keyboard::Keyboard(void) {
  // Use termios to turn off line buffering.
  static bool initialized = false;

  if (initialized) return;

  termios termios_;

  tcgetattr(0, &termios_);

  termios_.c_lflag &= ~static_cast<unsigned int>(ICANON);

  tcsetattr(0, TCSANOW, &termios_);

  setbuf(stdin, NULL);

  initialized = true;
}

bool Keyboard::trigger_key(char const val) {
  return this->_map_chars.find(val) != std::wstring::npos;
}

int Keyboard::_kbhit(void) {
  int bytes_waiting;

  ioctl(0, FIONREAD, &bytes_waiting);

  return bytes_waiting;
}

void Keyboard::clear_keys_pressed(void) { this->_map_chars.clear(); }

void Keyboard::collect_keys_pressed(void) {
  int n_bytes;

  if ((n_bytes = this->_kbhit()) == 0) return;

  wchar_t *buffers = new wchar_t[n_bytes];

  std::wcin.read(buffers, static_cast<std::streamsize>(n_bytes));

  this->_map_chars = std::wstring(buffers, static_cast<unsigned long>(n_bytes));

  delete[] (buffers);
}
}  // namespace DL::Term