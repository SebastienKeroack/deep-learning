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
#include "deep-learning/io/form/dialogbox.hpp"

// Deep learning:
#include "deep-learning/data/string.hpp"
#include "deep-learning/io/logger.hpp"

using namespace DL::Str;

namespace DL::Form {
std::string which_dialogbox(void) {
  // If system can use system command.
  if (::system(NULL) != 0)
    return "";
  else if (::system("which gdialog") == 0)
    return "gdialog";
  else if (::system("which kdialog") == 0)
    return "kdialog";

  return "";
}

bool spawn(std::wstring const &text, std::wstring const &title,
           char const *const type) {
  static std::string dialoger(which_dialogbox());
  if (dialoger.empty()) {
    ERR(L"There's no dialog available on the system.");
    return false;
  }

  std::string const cmd(dialoger + " --title \"" + to_string(title) + "\" --" +
                        type + " \"" + to_string(text) + "\"");

  return ::system(cmd.c_str()) == 0 ? true : false;
}

bool accept(std::wstring const &text, std::wstring const &title) {
  return spawn(text, title, "yesno");
}

bool ok(std::wstring const &text, std::wstring const &title) {
  return spawn(text, title, "msgbox");
}
}  // namespace DL::Form
