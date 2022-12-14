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

#pragma once

// Deep learning:
#include "deep-learning/data/enum/dialogbox.hpp"

// Standard:
#include <string>

namespace DL::Form {
bool accept(std::wstring const &text, std::wstring const &title);

bool dialog_box(DL::DIALOGBOX::TYPE const type, std::wstring const &text,
                std::wstring const &title);

bool ok(std::wstring const &text, std::wstring const &title);

#define DEBUG_BOX(text) DL::Form::dialog_box(DL::DIALOGBOX::OK, text, L"DEBUG");
}  // namespace DL::Form
