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
__host__ __device__ T *reallocate(T *ptr_array_received,
                                  size_t const new_size_t_received,
                                  size_t const old_size_t_received) {
  if (ptr_array_received == NULL) {
    return (NULL);
  } else if (new_size_t_received == old_size_t_received) {
    return (ptr_array_received);
  }

  T *tmp_ptr_array_T;

  if (CPY && old_size_t_received != 0_UZ) {
    tmp_ptr_array_T = static_cast<T *>(malloc(new_size_t_received));

    if (old_size_t_received < new_size_t_received) {
      if (SET) {
        memset(tmp_ptr_array_T + (old_size_t_received / sizeof(T)), 0,
               new_size_t_received - old_size_t_received);
      }

      memcpy(tmp_ptr_array_T, ptr_array_received, old_size_t_received);
    } else {
      memcpy(tmp_ptr_array_T, ptr_array_received, new_size_t_received);
    }

    free(ptr_array_received);
    ptr_array_received = NULL;
  } else {
    free(ptr_array_received);
    ptr_array_received = NULL;

    tmp_ptr_array_T = static_cast<T *>(malloc(new_size_t_received));

    if (SET) {
      memset(tmp_ptr_array_T, 0, new_size_t_received);
    }
  }

  return (tmp_ptr_array_T);
}
}  // namespace DL::Memory::C
