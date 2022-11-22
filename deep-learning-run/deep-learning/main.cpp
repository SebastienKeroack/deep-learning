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

// Adept source (include only once in the program):
#ifdef COMPILE_ADEPT
#pragma warning(push)
#pragma warning(disable : 4244)
#include <adept_source.h>
#pragma warning(pop)
#endif

// Deep learning:
#include "deep-learning/mnist.hpp"
#include "deep-learning/v1/custom.hpp"

// Deep learning lib:
#include "deep-learning-lib/data/dtypes.hpp"
#include "deep-learning-lib/io/term/input.hpp"
#include "deep-learning-lib/session.hpp"

// Standard:
#ifdef COMPILE_WINDOWS
#include <iostream>
#endif

#include <vector>

using namespace DL;
using namespace DL::Sys;
using namespace DL::Term;

// clang-format off
struct MENU {
  typedef enum : int {
    MNIST,
    QUIT,
    LENGTH
  } TYPE;
};
// clang-format on

#ifdef COMPILE_WINDOWS
int wmain(int const n_args, wchar_t const *const args[]) {
  if (SetConsoleTitleW(L"Menu - Deep learning") == FALSE)
    std::wcout << L"WARNING: Couldn't set a title to the console." << std::endl;
#if defined(_DEBUG) && defined(_CRTDBG_MAP_ALLOC)
  _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

  unsigned int control_word;
  _controlfp_s(&control_word, _EM_INEXACT | _EM_UNDERFLOW | _EM_OVERFLOW,
               _MCW_EM);
#endif
#elif COMPILE_LINUX
int main(int const n_args, char const *const args[]) {
#endif

  Session sess(n_args, args);

  INFO(L"******************************************");
  INFO(L"\tCopyright Sébastien Kéroack");
  INFO(L"");
  INFO(L"******************************************");
  INFO(L"");

#ifdef _DEBUG
  if (ARG_EXIST(L"--run-custom")) {
    run_custom();
    return EXIT_SUCCESS;
  }
#endif

  MENU::TYPE option(MENU::TYPE(0));
  while (true) {
    if (option < MENU::LENGTH)
      INFO(
          LR""""(
Main Menu:
  [0]: MNIST.
  [1]: Quit.
)"""");

    option =
        MENU::TYPE(parse_discrete(0, MENU::LENGTH - 1, L"Choose an option: "));

    if (option != MENU::QUIT) {
      INFO(L"");

#ifdef COMPILE_WINDOWS
      if (SetConsoleTitleW(L"Menu - Deep learning") == FALSE)
        WARN(L"Couldn't set a title to the console.");
#endif
    }

    switch (option) {
      case MENU::MNIST:
        if (run_mnist() == false)
          ERR(L"An error has been triggered from the `run_mnist()` function.");
        break;
      default:  // MENU::QUIT
        break;
    }

    if (option == MENU::QUIT || sess.preparing_for_shutdown()) break;
  }

  return EXIT_SUCCESS;
}
