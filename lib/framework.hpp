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

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN  // Exclude rarely-used stuff from Windows headers
#endif

#ifndef DEEPLEARNING_USE_ADEPT
#if defined(_DEBUG) && __has_include("adept.h")
#define DEEPLEARNING_USE_ADEPT 1
#else
#define DEEPLEARNING_USE_ADEPT 0
#endif
#endif

#ifndef DEEPLEARNING_USING_REAL_AS_DOUBLE // ELSE USE FLOAT
#ifdef _DEBUG
#define DEEPLEARNING_USING_REAL_AS_DOUBLE 1
#else
#define DEEPLEARNING_USING_REAL_AS_DOUBLE 0
#endif
#endif

#if defined(_CRTDBG_MAP_ALLOC) && !defined(__CUDA_ARCH__)
/* NOTE (https://msdn.microsoft.com/fr-ca/library/x98tx3cf.aspx):
  1. Overwriting the keyword `new` will break the compatibility to use
  boost::spirit::x3 unless the conformance mode is changed from
  `/permissive-` to `/permissive`.
  2. It will also break the compatibility with Eigen and changing the
  conformance mode is useless in this case.

  WORKAROUND: The keyword `new` w.r.t `CRTDBG_NEW` is undefined when including
  these headers and replaced by the original keyword `new`. */
#define CRTDBG_NEW new (_NORMAL_BLOCK, __FILE__, __LINE__)
#define new CRTDBG_NEW

#include <crtdbg.h>
#include <cstdlib>
#endif