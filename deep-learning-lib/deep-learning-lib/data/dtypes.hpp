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

// Adept:
#ifdef COMPILE_ADEPT
#include <adept.h>
#endif

// Standard:
#ifdef COMPILE_LINUX
#include <cstddef>
#endif

#include <limits>

#ifndef FAIL
inline constexpr int FAIL(-1);
#endif

#ifndef FALSE
inline constexpr int FALSE(0);
#endif

#ifndef NULL
inline constexpr int NULL(0);
#endif

#define SAFE_FREE(ptr) \
  if (ptr) {           \
    free(ptr);         \
    ptr = NULL;        \
  }

#define SAFE_DELETE(ptr) \
  delete (ptr);          \
  ptr = nullptr;

#define SAFE_DELETE_ARRAY(ptr) \
  delete[] (ptr);              \
  ptr = nullptr;

#ifndef TRUE
inline constexpr int TRUE(1);
#endif

#ifdef COMPILE_DOUBLE
using real = double;
#else
using real = float;
#endif

#ifdef COMPILE_ADEPT
using var = adept::Active<real>;

// Print an active scalar to a wstream
inline std::wostream& operator<<(std::wostream& os, const var& v) {
  os << v.value();
  return os;
}

inline real cast(var const& expr) { return expr.value(); }

#ifdef COMPILE_DOUBLE
inline double castd(var const& expr) { return expr.value(); }
#else
inline double castd(var const& expr) {
  return static_cast<double>(expr.value());
}
#endif

inline int casti(var const& expr) { return static_cast<int>(expr.value()); }
#else
using var = real;

inline real cast(var const &expr) { return expr; }

#ifdef COMPILE_DOUBLE
inline double castd(var const &expr) { return expr; }
#else
inline double castd(var const &expr) { return static_cast<double>(expr); }
#endif

inline int casti(var const &expr) { return static_cast<int>(expr); }
#endif

constexpr real operator""_r(unsigned long long int lhs) {
  return static_cast<real>(lhs);
}

constexpr real operator""_r(long double lhs) { return static_cast<real>(lhs); }

constexpr size_t operator""_UZ(unsigned long long int lhs) {
  return static_cast<size_t>(lhs);
}

inline constexpr real EPSILON(1e-7_r);
inline constexpr size_t KILOBYTE(1024);
inline constexpr size_t MEGABYTE(1048576);
