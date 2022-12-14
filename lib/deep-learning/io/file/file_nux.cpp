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
#include "deep-learning/data/string.hpp"

// Standard:
#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>

using namespace DL::Str;

namespace DL::File {
std::wstring home_directory(void) {
  char const *path_name;

  if ((path_name = getenv("HOME")) == NULL)
    path_name = getpwuid(getuid())->pw_dir;

  return to_wstring(path_name);
}
}  // namespace DL::File