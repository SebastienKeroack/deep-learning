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
#include "deep-learning/io/term/input.hpp"

// Deep learning:
#include "deep-learning/io/logger.hpp"

// Standard:
#ifdef __linux__
#include <termios.h>
#include <unistd.h>
#endif

#include <stdarg.h>

#include <iostream>
#include <regex>
#include <unordered_set>

namespace DL::Term {
wchar_t getwch(void) {
#ifdef _WIN32
  return _getwch();
#elif __linux__
  struct termios oldattr, newattr;
  tcgetattr(STDIN_FILENO, &oldattr);
  newattr = oldattr;
  newattr.c_lflag &= ~(static_cast<unsigned int>(ICANON | ECHO));
  tcsetattr(STDIN_FILENO, TCSANOW, &newattr);
  wint_t out(getwchar());
  tcsetattr(STDIN_FILENO, TCSANOW, &oldattr);
  return out == 127 ? L'\b' : static_cast<wchar_t>(out);
#endif
}

bool accept(wchar_t const *const text) {
  std::unordered_set<std::wstring> yes = {L"yes", L"ye", L"y"},
                                   no = {L"no", L"n"};

  int val;
  wchar_t inp;

  std::wstring line(L"");

  bool is_digit;
  bool ret;

  if (text != NULL) INFO(text);

  logger->new_line = false;
  INFO(L"Y̲es or N̲o: ");
  logger->new_line = true;

  do {
    if (line.empty() == false) {
      std::wcout << std::wstring(line.size(), L'\b');
      std::wcout << std::wstring(line.size(), L' ');
      std::wcout << std::wstring(line.size(), L'\b');
      line = L"";
    }

    is_digit = false;
    inp = getwch();
    while (std::wcin.fail() == false && inp != L'\r' && inp != L'\n') {
      if (inp == L'\b') {
        if (line.empty() == false) {
          std::wcout << std::wstring(L"\b \b");
          line.pop_back();
        }
      } else {
        if (isdigit(inp)) is_digit = true;
        std::wcout << inp;
        line += inp;
      }
      inp = getwch();
    }

    if (is_digit) {
      if (line.size() != 1_UZ) continue;

      val = std::stoi(line);

      if (val == 0 || val == 1) {
        ret = static_cast<bool>(val);
        break;
      }
    } else if (line.size() <= 3_UZ) {
      std::transform(line.begin(), line.end(), line.begin(), ::tolower);

      if (yes.contains(line)) {
        ret = true;
        break;
      } else if (no.contains(line)) {
        ret = false;
        break;
      }
    }
  } while (true);

  std::wcout << std::endl;

  return ret;
}

template <typename T>
T parse_discrete(T const minval, wchar_t const *const text) {
  return parse_discrete(minval, (std::numeric_limits<T>::max)(), text);
}

template <typename T>
T parse_discrete(T const minval, T const maxval, wchar_t const *const text) {
  T val(0);

  wchar_t inp;

  std::wstring line(L"");
  std::wsmatch matches;
  std::wregex re(L"^([-+]?[0-9]+)$");

  if (text != NULL) {
    logger->new_line = false;
    INFO(text);
    logger->new_line = true;
  }

  do {
    if (line.empty() == false) {
      std::wcout << std::wstring(line.size(), L'\b');
      std::wcout << std::wstring(line.size(), L' ');
      std::wcout << std::wstring(line.size(), L'\b');
      line = L"";
    }

    inp = getwch();
    while (std::wcin.fail() == false && inp != L'\r' && inp != L'\n') {
      if (inp == L'\b') {
        if (line.empty() == false) {
          std::wcout << std::wstring(L"\b \b");
          line.pop_back();
        }
      } else {
        std::wcout << inp;
        line += inp;
      }
      inp = getwch();
    }

    if (std::regex_match(line, matches, re,
                         std::regex_constants::match_default) == false)
      continue;

    try {
      if constexpr (std::is_same<T, int>::value)
        val = std::stoi(matches[1]);
      else if constexpr (std::is_same<T, long>::value)
        val = std::stol(matches[1]);
      else if constexpr (std::is_same<T, long long>::value)
        val = std::stoll(matches[1]);
      else if constexpr (std::is_same<T, unsigned int>::value)
        val = static_cast<unsigned int>(std::stoul(matches[1]));
      else if constexpr (std::is_same<T, unsigned long>::value)
        val = std::stoul(matches[1]);
      else if constexpr (std::is_same<T, unsigned long long>::value)
        val = std::stoull(matches[1]);
      else
        throw std::logic_error("NotImplementedException");

      if (val < minval || val > maxval) continue;
      break;
    } catch (...) {
      continue;
    }
  } while (true);

  std::wcout << std::endl;

  return val;
}

template <typename T>
T parse_real(T const minval, wchar_t const *const text) {
  return parse_real(minval, (std::numeric_limits<T>::max)(), text);
}

template <typename T>
T parse_real(T const minval, T const maxval, wchar_t const *const text) {
  T val(0);

  wchar_t inp;

  std::wstring line(L"");
  std::wsmatch matches;
  std::wregex re(L"^([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?)$");

  if (text != NULL) {
    logger->new_line = false;
    INFO(text);
    logger->new_line = true;
  }

  do {
    if (line.empty() == false) {
      std::wcout << std::wstring(line.size(), L'\b');
      std::wcout << std::wstring(line.size(), L' ');
      std::wcout << std::wstring(line.size(), L'\b');
      line = L"";
    }

    inp = getwch();
    while (std::wcin.fail() == false && inp != L'\r' && inp != L'\n') {
      if (inp == L'\b') {
        if (line.empty() == false) {
          std::wcout << std::wstring(L"\b \b");
          line.pop_back();
        }
      } else {
        std::wcout << inp;
        line += inp;
      }
      inp = getwch();
    }

    if (std::regex_match(line, matches, re,
                         std::regex_constants::match_default) == false)
      continue;

    try {
      if constexpr (std::is_same<T, float>::value)
        val = std::stof(matches[1]);
      else if constexpr (std::is_same<T, double>::value)
        val = std::stod(matches[1]);
      else if constexpr (std::is_same<T, double long>::value)
        val = std::stold(matches[1]);
      else
        throw std::logic_error("NotImplementedException");

      if (val < minval || val > maxval) continue;
      break;
    } catch (...) {
      continue;
    }
  } while (true);

  std::wcout << std::endl;

  return val;
}

std::wstring parse_wstring(wchar_t const *const text) {
  if (text != NULL) {
    logger->new_line = false;
    INFO(text);
    logger->new_line = true;
  }

  std::wstring out(L"");
  wchar_t inp(getwch());
  while (std::wcin.fail() == false && inp != L'\r' && inp != L'\n') {
    if (inp == L'\b') {
      if (out.empty() == false) {
        std::wcout << std::wstring(L"\b \b");
        out.pop_back();
      }
    } else {
      std::wcout << inp;
      out += inp;
    }
    inp = getwch();
  }

  std::wcout << std::endl;

  return out;
}

void pause(void) {
#ifdef _WIN32
  ::system("PAUSE");
#else
  std::wcout << L"Press \"ENTER\" key to continue." << std::endl;
  std::wcin.get();
#endif
}
}  // namespace DL::Term

// clang-format off
template int DL::Term::parse_discrete(int const, wchar_t const *const);
template long DL::Term::parse_discrete<long>(long const, wchar_t const *const);
template long long DL::Term::parse_discrete(long long const, wchar_t const *const);
template unsigned int DL::Term::parse_discrete<unsigned int>(unsigned int const, wchar_t const *const);
template unsigned long DL::Term::parse_discrete<unsigned long>(unsigned long const, wchar_t const *const);
template unsigned long long DL::Term::parse_discrete<unsigned long long>(unsigned long long const, wchar_t const *const);

template int DL::Term::parse_discrete(int const, int const, wchar_t const *const);
template long DL::Term::parse_discrete<long>(long const, long const, wchar_t const *const);
template long long DL::Term::parse_discrete(long long const, long long const, wchar_t const *const);
template unsigned int DL::Term::parse_discrete<unsigned int>(unsigned int const, unsigned int const, wchar_t const *const);
template unsigned long DL::Term::parse_discrete<unsigned long>(unsigned long const, unsigned long const, wchar_t const *const);
template unsigned long long DL::Term::parse_discrete<unsigned long long>(unsigned long long const, unsigned long long const, wchar_t const *const);

template float DL::Term::parse_real<float>(float const, wchar_t const *const);
template double DL::Term::parse_real<double>(double const, wchar_t const *const);
template double long DL::Term::parse_real<double long>(double long const, wchar_t const *const);

template float DL::Term::parse_real<float>(float const, float const, wchar_t const *const);
template double DL::Term::parse_real<double>(double const, double const, wchar_t const *const);
template double long DL::Term::parse_real<double long>(double long const, double long const, wchar_t const *const);
// clang-format on