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
#include "deep-learning/io/logger.hpp"

namespace DL::Form {
bool dialog_box(DIALOGBOX::TYPE const type, std::wstring const &text,
                std::wstring const &title) {
  switch (type) {
    case DIALOGBOX::ACCEPT:
      return accept(text, title);
    case DIALOGBOX::OK:
      return ok(text, title);
    default:
      ERR(L"The `%ls` dialog box type is not supported in the switch.",
          DIALOGBOX_NAME[type].c_str());
      return false;
  }
}
}  // namespace DL::Form
