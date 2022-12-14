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

#if defined(_DEBUG) && defined(_WIN32)
#include <iterator>
#endif

namespace DL::v1::Mem {
template <class T, bool STD>
__host__ __device__ void copy(T const *ptr_array_source_received,
                              T const *ptr_array_last_source_received,
                              T *ptr_array_destination_received) {
  if (STD) {
#ifdef __CUDA_ARCH__
    asm("trap;");
#else
#if defined(_DEBUG) && defined(_WIN32)
    std::copy(ptr_array_source_received, ptr_array_last_source_received,
              stdext::checked_array_iterator<T *>(
                  ptr_array_destination_received,
                  static_cast<size_t>(ptr_array_last_source_received -
                                      ptr_array_source_received)));
#else
    std::copy(ptr_array_source_received, ptr_array_last_source_received,
              ptr_array_destination_received);
#endif
#endif
  } else {
    while (ptr_array_source_received != ptr_array_last_source_received) {
      *ptr_array_destination_received++ = *ptr_array_source_received++;
    }
  }
}

template <class T, bool STD>
__host__ __device__ void fill(T *ptr_array_received,
                              T *const ptr_array_last_received,
                              T const value_received) {
  if (STD) {
#ifdef __CUDA_ARCH__
    asm("trap;");
#else
    std::fill(ptr_array_received, ptr_array_last_received, value_received);
#endif
  } else {
    while (ptr_array_received != ptr_array_last_received) {
      *ptr_array_received++ = value_received;
    }
  }
}

template <class T>
__host__ __device__ void fill_null(T *ptr_array_received,
                                      T const *const ptr_array_last_received) {
  while (ptr_array_received != ptr_array_last_received) {
    *ptr_array_received++ = nullptr;
  }
}

template <class T, bool CPY, bool SET>
__host__ __device__ T *reallocate(T *ptr_array_received,
                                  size_t const new_size_received,
                                  size_t const old_size_received) {
  if (ptr_array_received == nullptr) {
    return nullptr;
  } else if (new_size_received == old_size_received) {
    return (ptr_array_received);
  }

  T *tmp_ptr_array_T;

  if (CPY && old_size_received != 0_UZ) {
    tmp_ptr_array_T = new T[new_size_received];

    if (old_size_received < new_size_received) {
      if (SET) {
        DL::Memory::fill<T, defined(__CUDA_ARCH__)>(
            tmp_ptr_array_T + old_size_received,
            tmp_ptr_array_T + new_size_received, T(0));
      }

      DL::Memory::copy<T, defined(__CUDA_ARCH__)>(
          ptr_array_received, ptr_array_received + old_size_received,
          tmp_ptr_array_T);
    } else {
      DL::Memory::copy<T, defined(__CUDA_ARCH__)>(
          ptr_array_received, ptr_array_received + new_size_received,
          tmp_ptr_array_T);
    }

    delete[](ptr_array_received);
    ptr_array_received = nullptr;
  } else {
    delete[](ptr_array_received);
    ptr_array_received = nullptr;

    tmp_ptr_array_T = new T[new_size_received];

    if (SET) {
      DL::Memory::fill<T, defined(__CUDA_ARCH__)>(
          tmp_ptr_array_T, tmp_ptr_array_T + new_size_received, T(0));
    }
  }

  return (tmp_ptr_array_T);
}

template <class T, bool CPY, bool SET>
__device__ T *reallocate(T *ptr_array_received, size_t const new_size_received,
                         size_t const old_size_received,
                         struct dim3 const &ref_dimension_grid_set_received,
                         struct dim3 const &ref_dimension_block_set_received,
                         struct dim3 const &ref_dimension_grid_cpy_received,
                         struct dim3 const &ref_dimension_block_cpy_received) {
  if (ptr_array_received == nullptr) {
    return nullptr;
  } else if (new_size_received == old_size_received) {
    return (ptr_array_received);
  }

  T *tmp_ptr_array_T;

  if (CPY && old_size_received != 0_UZ) {
    tmp_ptr_array_T = new T[new_size_received];

    if (old_size_received < new_size_received) {
      if (SET) {
        Zero_1D<T>(new_size_received - old_size_received,
                   tmp_ptr_array_T + old_size_received,
                   ptr_dimension_grid_zero_received,
                   ptr_dimension_block_zero_received);
      }

      DL::Memory::Memory_Copy_1D<T>(
          old_size_received, tmp_ptr_array_T, ptr_array_received,
          ptr_dimension_grid_copy_received, ptr_dimension_block_copy_received);

      // Do we need to synchronise? Based on "Memory_Copy_1D" Function.
      // => Synchronisation before deleting the old array.
      if (old_size_received >= warpSize * warpSize) {
        cudaDeviceSynchronize();
      }
    } else {
      DL::Memory::Memory_Copy_1D<T>(
          new_size_received, tmp_ptr_array_T, ptr_array_received,
          ptr_dimension_grid_copy_received, ptr_dimension_block_copy_received);

      // Do we need to synchronise? Based on "Memory_Copy_1D" Function.
      // => Synchronisation before deleting the old array.
      if (new_size_received >= warpSize * warpSize) {
        cudaDeviceSynchronize();
      }
    }

    delete[](ptr_array_received);
    ptr_array_received = nullptr;
  } else {
    delete[](ptr_array_received);
    ptr_array_received = nullptr;

    tmp_ptr_array_T = new T[new_size_received];

    if (SET) {
      Zero_1D<T>(new_size_received, tmp_ptr_array_T,
                 ptr_dimension_grid_zero_received,
                 ptr_dimension_block_zero_received);

      // Do we need to synchronise? Based on "Zero_1D" Function.
      // => Synchronisation before deleting the old array.
      if (new_size_received >= warpSize * warpSize) {
        cudaDeviceSynchronize();
      }
    }
  }

  return (tmp_ptr_array_T);
}

template <class T, bool CPY>
__host__ __device__ T *reallocate_obj(T *ptr_array_received,
                                          size_t const new_size_received,
                                          size_t const old_size_received) {
  if (ptr_array_received == nullptr) {
    return nullptr;
  } else if (new_size_received == old_size_received) {
    return (ptr_array_received);
  }

  T *tmp_ptr_array_T;

  if (CPY && old_size_received != 0_UZ) {
    tmp_ptr_array_T = new T[new_size_received];

    if (old_size_received < new_size_received) {
      DL::Memory::copy<T, defined(__CUDA_ARCH__)>(
          ptr_array_received, ptr_array_received + old_size_received,
          tmp_ptr_array_T);
    } else {
      DL::Memory::copy<T, defined(__CUDA_ARCH__)>(
          ptr_array_received, ptr_array_received + new_size_received,
          tmp_ptr_array_T);
    }

    delete[](ptr_array_received);
    ptr_array_received = nullptr;
  } else {
    delete[](ptr_array_received);
    ptr_array_received = nullptr;

    tmp_ptr_array_T = new T[new_size_received];
  }

  return (tmp_ptr_array_T);
}

template <class T, bool CPY>
__device__ T *reallocate_obj(
    T *ptr_array_received, size_t const new_size_received,
    size_t const old_size_received,
    struct dim3 const &ref_dimension_grid_set_received,
    struct dim3 const &ref_dimension_block_set_received,
    struct dim3 const &ref_dimension_grid_cpy_received,
    struct dim3 const &ref_dimension_block_cpy_received) {
  if (ptr_array_received == nullptr) {
    return nullptr;
  } else if (new_size_received == old_size_received) {
    return (ptr_array_received);
  }

  T *tmp_ptr_array_T;

  if (CPY && old_size_received != 0_UZ) {
    tmp_ptr_array_T = new T[new_size_received];

    if (old_size_received < new_size_received) {
      DL::Memory::Memory_Copy_1D<T>(
          old_size_received, tmp_ptr_array_T, ptr_array_received,
          ptr_dimension_grid_copy_received, ptr_dimension_block_copy_received);

      // Do we need to synchronise? Based on "Memory_Copy_1D" Function.
      // => Synchronisation before deleting the old array.
      if (old_size_received >= warpSize * warpSize) {
        cudaDeviceSynchronize();
      }
    } else {
      DL::Memory::Memory_Copy_1D<T>(
          new_size_received, tmp_ptr_array_T, ptr_array_received,
          ptr_dimension_grid_copy_received, ptr_dimension_block_copy_received);

      // Do we need to synchronise? Based on "Memory_Copy_1D" Function.
      // => Synchronisation before deleting the old array.
      if (new_size_received >= warpSize * warpSize) {
        cudaDeviceSynchronize();
      }
    }

    delete[](ptr_array_received);
    ptr_array_received = nullptr;
  } else {
    delete[](ptr_array_received);
    ptr_array_received = nullptr;

    tmp_ptr_array_T = new T[new_size_received];
  }

  return (tmp_ptr_array_T);
}

template <class T, bool CPY, bool SET>
__host__ __device__ T *reallocate_ptofpt(T *ptr_array_received,
                                         size_t const new_size_received,
                                         size_t const old_size_received) {
  if (ptr_array_received == nullptr) {
    return nullptr;
  } else if (new_size_received == old_size_received) {
    return (ptr_array_received);
  }

  T *tmp_ptr_array_T;

  if (CPY && old_size_received != 0_UZ) {
    tmp_ptr_array_T = new T[new_size_received];

    if (old_size_received < new_size_received) {
      if (SET) {
        fill_null<T>(tmp_ptr_array_T + old_size_received,
                        tmp_ptr_array_T + new_size_received);
      }

      // TODO: Check if std is compatible.
      DL::Memory::copy<T, false>(ptr_array_received,
                                   ptr_array_received + old_size_received,
                                   tmp_ptr_array_T);
    } else {
      // TODO: Check if std is compatible.
      DL::Memory::copy<T, false>(ptr_array_received,
                                   ptr_array_received + new_size_received,
                                   tmp_ptr_array_T);
    }

    delete[](ptr_array_received);
    ptr_array_received = nullptr;
  } else {
    delete[](ptr_array_received);
    ptr_array_received = nullptr;

    tmp_ptr_array_T = new T[new_size_received];

    if (SET) {
      fill_null<T>(tmp_ptr_array_T, tmp_ptr_array_T + new_size_received);
    }
  }

  return (tmp_ptr_array_T);
}

template <class T, bool CPY, bool SET>
__device__ T *reallocate_ptofpt(
    T *ptr_array_received, size_t const new_size_received,
    size_t const old_size_received,
    struct dim3 const &ref_dimension_grid_set_received,
    struct dim3 const &ref_dimension_block_set_received,
    struct dim3 const &ref_dimension_grid_cpy_received,
    struct dim3 const &ref_dimension_block_cpy_received) {
  if (ptr_array_received == nullptr) {
    return nullptr;
  } else if (new_size_received == old_size_received) {
    return (ptr_array_received);
  }

  T *tmp_ptr_array_T;

  if (CPY && old_size_received != 0_UZ) {
    tmp_ptr_array_T = new T[new_size_received];

    if (old_size_received < new_size_received) {
      if (SET) {
        Fill_Pointers_1D<T>(new_size_received - old_size_received,
                            tmp_ptr_array_T + old_size_received,
                            ptr_dimension_grid_fill_received,
                            ptr_dimension_block_fill_received);
      }

      DL::Memory::Memory_Copy_1D<T>(
          old_size_received, tmp_ptr_array_T, ptr_array_received,
          ptr_dimension_grid_copy_received, ptr_dimension_block_copy_received);

      // Do we need to synchronise? Based on "Memory_Copy_1D" Function.
      // => Synchronisation before deleting the old array.
      if (old_size_received >= warpSize * warpSize) {
        cudaDeviceSynchronize();
      }
    } else {
      DL::Memory::Memory_Copy_1D<T>(
          new_size_received, tmp_ptr_array_T, ptr_array_received,
          ptr_dimension_grid_copy_received, ptr_dimension_block_copy_received);

      // Do we need to synchronise? Based on "Memory_Copy_1D" Function.
      // => Synchronisation before deleting the old array.
      if (new_size_received >= warpSize * warpSize) {
        cudaDeviceSynchronize();
      }
    }

    delete[](ptr_array_received);
    ptr_array_received = nullptr;
  } else {
    delete[](ptr_array_received);
    ptr_array_received = nullptr;

    tmp_ptr_array_T = new T[new_size_received];

    if (SET) {
      Fill_Pointers_1D<T>(new_size_received, tmp_ptr_array_T,
                          ptr_dimension_grid_fill_received,
                          ptr_dimension_block_fill_received);

      // Do we need to synchronise? Based on "Zero_1D" Function.
      // => Synchronisation before deleting the old array.
      if (new_size_received >= warpSize * warpSize) {
        cudaDeviceSynchronize();
      }
    }
  }

  return (tmp_ptr_array_T);
}
}  // namespace DL::Memory
