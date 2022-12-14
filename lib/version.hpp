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

#define STRINGIZE(expr) #expr
#define STR(expr) STRINGIZE(expr)

#define WSTRINGIZE(expr) L## #expr
#define WSTR(expr) WSTRINGIZE(expr)

#define PV_MAJOR 0
#define PV_MINOR 2
#define PV_BUILD 22347
#define PV_REV 1

#define PRODUCT_VER_STRING \
  STR(PV_MAJOR) "." \
  STR(PV_MINOR) "." \
  STR(PV_BUILD) "." \
  STR(PV_REV)

#define PRODUCT_VER_WSTRING \
  WSTR(PV_MAJOR) L"." \
  WSTR(PV_MINOR) L"." \
  WSTR(PV_BUILD) L"." \
  WSTR(PV_REV)
