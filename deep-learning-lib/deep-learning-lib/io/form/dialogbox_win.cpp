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
#include "deep-learning-lib/pch.hpp"

// File header:
#include "deep-learning-lib/io/form/dialogbox.hpp"

// Standard:
#include <windows.h>

#include <string>

namespace DL::Form {
int spawn(std::wstring const &text, std::wstring const &title,
          UINT const type) {
  return MessageBoxW(NULL, text.c_str(), title.c_str(),
                     MB_ICONINFORMATION | type);
}

bool accept(std::wstring const &text, std::wstring const &title) {
  switch (spawn(text, title, MB_YESNO)) {
    case IDYES:
      return true;
    case IDNO:
      return false;
    default:
      return false;
  }
}

bool ok(std::wstring const &text, std::wstring const &title) {
  switch (spawn(text, title, MB_OK)) {
    case IDOK:
      return true;
    default:
      return true;
  }
}
}  // namespace DL::Form
