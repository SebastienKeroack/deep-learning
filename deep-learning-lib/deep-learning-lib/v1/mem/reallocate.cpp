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

#if defined(_DEBUG) && defined(COMPILE_WINDOWS)
#include <iterator>
#endif

namespace DL::v1::Mem {
template <class SRC, class DST>
void copy(SRC const *src, SRC const *const src_end, DST *dst) {
  if constexpr (std::is_same<SRC, DST>::value)
#if defined(_DEBUG) && defined(COMPILE_WINDOWS)
    std::copy(src, src_end,
              stdext::checked_array_iterator<SRC *>(
                  dst, static_cast<size_t>(src_end - src)));
#else
    std::copy(src, src_end, dst);
#endif
#ifdef COMPILE_ADEPT
  else if constexpr (std::is_same<SRC, var>::value)
    while (src != src_end) *dst++ = (*src++).value();
#endif
  else
    while (src != src_end) *dst++ = *src++;
}

template <class T, bool STD>
void copy(T const *src, T const *src_end, T *dst) {
  if (STD)
#if defined(_DEBUG) && defined(COMPILE_WINDOWS)
    std::copy(src, src_end,
              stdext::checked_array_iterator<T *>(
                  dst, static_cast<size_t>(src_end - src)));
#else
    std::copy(src, src_end, dst);
#endif
  else
    while (src != src_end) *dst++ = *src++;
}

template <class T, bool STD>
void fill(T *inp, T *const inp_end, T const val) {
  if (STD)
    std::fill(inp, inp_end, val);
  else
    while (inp != inp_end) *inp++ = val;
}

template <class T>
void fill_null(T *inp, T const *const inp_end) {
  while (inp != inp_end) *inp++ = nullptr;
}

template <class T, bool CPY, bool SET>
T *reallocate(T *inp, size_t const new_size, size_t const old_size) {
  if (inp == nullptr)
    return nullptr;
  else if (new_size == old_size)
    return inp;

  T *out;

  if (CPY && old_size != 0_UZ) {
    out = new T[new_size];

    if (old_size < new_size) {
      if (SET) fill<T>(out + old_size, out + new_size, T(0));

      copy<T>(inp, inp + old_size, out);
    } else {
      copy<T>(inp, inp + new_size, out);
    }

    delete[] (inp);
    inp = nullptr;
  } else {
    delete[] (inp);
    inp = nullptr;

    out = new T[new_size];

    if (SET) fill<T>(out, out + new_size, T(0));
  }

  return out;
}

template <class T, bool CPY>
T *reallocate_obj(T *inp, size_t const new_size, size_t const old_size) {
  if (inp == nullptr)
    return nullptr;
  else if (new_size == old_size)
    return (inp);

  T *out;

  if (CPY && old_size != 0_UZ) {
    out = new T[new_size];

    if (old_size < new_size)
      copy<T>(inp, inp + old_size, out);
    else
      copy<T>(inp, inp + new_size, out);

    delete[] (inp);
    inp = nullptr;
  } else {
    delete[] (inp);
    inp = nullptr;

    out = new T[new_size];
  }

  return out;
}

template <class T, bool CPY, bool SET>
T *reallocate_ptofpt(T *inp, size_t const new_size, size_t const old_size) {
  if (inp == nullptr)
    return nullptr;
  else if (new_size == old_size)
    return (inp);

  T *out;

  if (CPY && old_size != 0_UZ) {
    out = new T[new_size];

    if (old_size < new_size) {
      if (SET) fill_null<T>(out + old_size, out + new_size);

      copy<T, false>(inp, inp + old_size, out);
    } else {
      copy<T, false>(inp, inp + new_size, out);
    }

    delete[] (inp);
    inp = nullptr;
  } else {
    delete[] (inp);
    inp = nullptr;

    out = new T[new_size];

    if (SET) fill_null<T>(out, out + new_size);
  }

  return out;
}
}  // namespace DL::v1::Mem