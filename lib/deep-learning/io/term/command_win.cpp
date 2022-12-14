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
#include "pch.hpp"

// File header:
#include "deep-learning/io/term/command.hpp"

// Deep learning:
#include "deep-learning/io/file.hpp"
#include "deep-learning/io/logger.hpp"

// FMT:
#include <fmt/core.h>
#include <fmt/xchar.h>

using namespace DL::File;

namespace DL::Term {
std::wstring check_output(wchar_t const *const cmd) {
  FILE *file;

  if ((file = _wpopen(cmd, L"r")) == NULL) {
    ERR(L"An error has been triggered from the `popen(%ls)` function.", cmd);
    return L"";
  }

  wchar_t buff[1024];
  std::wstring output(L"");
  while (fgetws(buff, sizeof(buff), file) != NULL) output += buff;

  if (ferror(file) != 0)
    ERR(L"An error has been triggered from the `fgets(%ls, %zu, %ls)` "
        L"function.",
        buff, sizeof(buff), cmd);

  if (_pclose(file) == -1)
    ERR(L"An error has been triggered from the `pclose(%ls)` function.", cmd);

  return output;
}

bool download(wchar_t const *const uri, wchar_t const *const out) {
  if (path_exist(out)) return true;

  std::wstring const cmd_(fmt::format(
      L"start powershell.exe Invoke-WebRequest -Uri {} -OutFile {}", uri, out));
  wchar_t const *const cmd = cmd_.c_str();
  FILE *file;

  INFO(L"%ls", cmd);
  if ((file = _wpopen(cmd, L"r")) == NULL) {
    ERR(L"An error has been triggered from the `popen(%ls)` function.", cmd);
    return false;
  }

  // Exhaust output.
  wchar_t buff[1024];
  while (fgetws(buff, sizeof(buff), file) != NULL)
    ;

  if (_pclose(file) == -1) {
    ERR(L"An error has been triggered from the `pclose(%ls)` function.", cmd);
    return false;
  }

  if (path_exist(out) == false) {
    ERR(L"The requested file `%ls` has not been downloaded from `%ls`.", out,
        uri);
    return false;
  }

  return true;
}

bool execute(wchar_t const *const cmd) {
  FILE *file;

  INFO(L"%ls", cmd);
  if ((file = _wpopen(cmd, L"r")) == NULL) {
    ERR(L"An error has been triggered from the `popen(%ls)` function.", cmd);
    return false;
  }

  // Exhaust output.
  wchar_t buff[1024];
  while (fgetws(buff, sizeof(buff), file) != NULL)
    ;

  if (_pclose(file) == -1) {
    ERR(L"An error has been triggered from the `pclose(%ls)` function.", cmd);
    return false;
  }

  return true;
}

bool gunzip(wchar_t const *const path_name) {
  if (path_exist(path_name)) return true;

  std::wstring const path_name_gz(fmt::format(L"{}.gz", path_name).c_str());
  if (path_exist(path_name_gz) == false) {
    ERR(L"No such file to unzip `%ls`.", path_name_gz.c_str());
    return false;
  }

  std::wstring cmd_(
      fmt::format(L"start /b \"\" \"C:\\Program Files\\Git\\bin\\bash.exe\" "
                  L"gunzip \"{}\"",
                  path_name_gz));
  wchar_t const *const cmd = cmd_.c_str();
  FILE *file;

  INFO(L"%ls", cmd);
  if ((file = _wpopen(cmd, L"r")) == NULL) {
    ERR(L"An error has been triggered from the `popen(%ls)` function.", cmd);
    return false;
  }

  // Exhaust output.
  wchar_t buff[1024];
  while (fgetws(buff, sizeof(buff), file) != NULL)
    ;

  if (_pclose(file) == -1) {
    ERR(L"An error has been triggered from the `pclose(%ls)` function.", cmd);
    return false;
  }

  if (path_exist(path_name) == false) {
    ERR(L"The requested file `%ls` has not been unzip.", path_name);
    return false;
  }

  return true;
}
}  // namespace DL::Term