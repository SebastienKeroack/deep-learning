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

namespace DL::v1::Mem::C {
template <class T, bool CPY, bool SET>
T *reallocate(T *inp, size_t const new_size, size_t const old_size) {
  if (inp == NULL)
    return NULL;
  else if (new_size == old_size)
    return inp;

  T *out;

  if (CPY && old_size != 0_UZ) {
    out = static_cast<T *>(malloc(new_size));

    if (old_size < new_size) {
      if (SET) memset(out + (old_size / sizeof(T)), 0, new_size - old_size);

      memcpy(out, inp, old_size);
    } else {
      memcpy(out, inp, new_size);
    }

    free(inp);
    inp = NULL;
  } else {
    free(inp);
    inp = NULL;

    out = static_cast<T *>(malloc(new_size));

    if (SET) memset(out, 0, new_size);
  }

  return out;
}
}  // namespace DL::v1::Mem::C
