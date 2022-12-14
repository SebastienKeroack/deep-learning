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

#pragma once

#include "../lib/version.hpp"

#define FV_MAJOR PV_MAJOR
#define FV_MINOR PV_MINOR
#define FV_BUILD 22347
#define FV_REV 1

#define FILE_VER_STRING \
  STR(FV_MAJOR) "." \
  STR(FV_MINOR) "." \
  STR(FV_BUILD) "." \
  STR(FV_REV)

#define ORIGINAL_FILE_NAME \
  "test_" STR(FV_MAJOR) "-" STR(FV_MINOR) "_x64.exe"
