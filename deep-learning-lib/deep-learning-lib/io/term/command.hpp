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

// Standard:
#include <string>

namespace DL::Term {
std::wstring check_output(wchar_t const *const cmd);
bool download(wchar_t const *const uri, wchar_t const *const out);
bool execute(wchar_t const *const cmd);
bool gunzip(wchar_t const *const path_name);
}  // namespace DL::Term