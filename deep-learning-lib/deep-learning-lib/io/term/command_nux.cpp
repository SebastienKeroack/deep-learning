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

// PCH:
#include "deep-learning-lib/pch.hpp"

// File header:
#include "deep-learning-lib/io/term/command.hpp"

// Deep learning lib:
#include "deep-learning-lib/io/logger.hpp"

namespace DL::Term {
std::wstring check_output(wchar_t const *const cmd) {
  FILE *file;

  if ((file = popen(cmd, L"r")) == NULL) {
    ERR(L"An error has been triggered from the `popen(%ls)` function.", cmd);
    return L"";
  }

  char buff[1024];
  std::string output("");
  while (fgets(buff, sizeof(buff), file) != NULL) output += buff;

  if (ferror(file) != 0)
    ERR(L"An error has been triggered from the `fgets(%ls, %zu, %ls)` "
        L"function.",
        buff, sizeof(buff), cmd);

  if (pclose(file) == -1)
    ERR(L"An error has been triggered from the `pclose(%ls)` function.", cmd);

  return output;
}

bool execute(wchar_t const *const cmd) {
  FILE *file;

  if ((file = popen(cmd, L"r")) == NULL) {
    ERR(L"An error has been triggered from the `popen(%ls)` function.", cmd);
    return false;
  }

  // Exhaust output.
  char buff[1024];
  while (fgets(buff, sizeof(buff), file) != NULL)
    ;

  if (pclose(file) == -1) {
    ERR(L"An error has been triggered from the `pclose(%ls)` function.", cmd);
    return false;
  }

  return true;
}
}  // namespace DL::Term