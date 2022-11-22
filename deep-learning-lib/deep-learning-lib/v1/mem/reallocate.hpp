/* Copyright 2016, 2019 Sébastien Kéroack. All Rights Reserved.

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

namespace DL::v1::Mem {
#ifdef COMPILE_ADEPT
#define VARZERO(x, sz)    DL::v1::Mem::fill<var>(x, x + (sz / sizeof(var)), var(0))
#define VARCOPY(d, s, sz) DL::v1::Mem::copy<var>(s, s + (sz / sizeof(var)), d)
#else
#define VARZERO(x, sz)    memset(x, 0, sz)
#define VARCOPY(d, s, sz) memcpy(d, s, sz)
#endif

template <class SRC, class DST>
void copy(SRC const *src, SRC const *const src_end, DST *dst);

template <class T, bool STD = true>
void copy(T const *ptr_array_source_received,
          T const *ptr_array_last_source_received,
          T *ptr_array_destination_received);

template <class T, bool STD = true>
void fill(T *ptr_array_received, T *const ptr_array_last_received,
          T const value_received);

template <class T>
void fill_null(T *ptr_array_received, T const *const ptr_array_last_received);

template <class T, bool CPY = true, bool SET = true>
T *reallocate(T *ptr_array_received, size_t const new_size_received,
              size_t const old_size_received);

template <class T, bool CPY = true>
T *reallocate_obj(T *ptr_array_received, size_t const new_size_received,
                  size_t const old_size_received);

template <class T, bool CPY = true, bool SET = true>
T *reallocate_ptofpt(T *ptr_array_received, size_t const new_size_received,
                     size_t const old_size_received);
}  // namespace DL::v1::Mem

#include "deep-learning-lib/v1/mem/reallocate.cpp"
