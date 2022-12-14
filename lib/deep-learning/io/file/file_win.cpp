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
#include "deep-learning/io/file.hpp"

// Deep learning:
#include "deep-learning/io/logger.hpp"

// Windows.0:
#include <ShlObj.h>
// Windows.1:
#include <KnownFolders.h>
#include <initguid.h>

namespace DL::File {
std::wstring home_directory(void) {
  PWSTR path_name_ = NULL;

  if (FAILED(SHGetKnownFolderPath(FOLDERID_Documents, 0, NULL, &path_name_)))
    ERR(L"An error has been triggered from the `SHGetKnownFolderPath()` "
        "function.");

  std::wstring const path_name(path_name_);

  CoTaskMemFree(path_name_);

  return path_name;
}
}  // namespace DL::File