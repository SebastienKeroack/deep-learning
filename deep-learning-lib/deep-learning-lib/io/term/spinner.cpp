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
#include "deep-learning-lib/pch.hpp"

// File header:
#include "deep-learning-lib/io/term/spinner.hpp"

// Deep learning lib:
#include "deep-learning-lib/io/logger.hpp"

// Standard:
#include <iostream>
#include <stdexcept>
#include <string>

namespace DL::Term {
Spinner::Spinner(void) {}

Spinner::~Spinner(void) {}

void Spinner::join(void) {
  if (this->_thread.joinable() == false) {
    return;
  }

  this->_alive.store(false);
  this->_thread.join();
}

void Spinner::print(void) {
  // Print corresponding pattern.
  switch (this->_state) {
    case 0:
      std::wcout << L"[ / ]";
      break;
    case 1:
      std::wcout << L"[-]";
      break;
    case 2:
      std::wcout << L"[\\]";
      break;
    default:
      throw std::overflow_error(std::string(__FILE__) + ":" +
                                std::to_string(__LINE__));
  }

  if (++this->_state == 3) this->_state = 0;
}

void Spinner::run(void) {
  while (this->_alive.load()) {
    // Print corresponding pattern.
    this->print();

    // Sleep for smooth animation.
    std::this_thread::sleep_for(std::chrono::milliseconds(125));

    // Cursor go back 3 cases.
    std::wcout << std::wstring(3, '\b');
  }

  // clear waiting characters.
  std::wcout << std::wstring(3, ' ');
}

void Spinner::start(wchar_t const *const text) {
  if (this->_thread.joinable()) return;

  if (text != NULL) INFO(text);

  this->_alive.store(true);
  this->_thread = std::thread(&Spinner::run, this);
}
}  // namespace DL::Term
