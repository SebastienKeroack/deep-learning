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
#include "deep-learning/io/term/keyboard_win.hpp"

// Standard:
#include <windows.h>

namespace DL::Term {
Keyboard::Keyboard(void) {}

bool Keyboard::trigger_key(short const val) {
  if (GetAsyncKeyState(val) != 0)
    return GetConsoleWindow() == GetForegroundWindow();
  return false;
}
}  // namespace DL::Term