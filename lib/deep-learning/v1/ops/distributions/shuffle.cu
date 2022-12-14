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

#include <curand_kernel.h>

namespace DL::v1::Dist {
template <typename T>
__global__ void kernel__Tree_Shift_Shuffle(
    size_t const half_size_floor_received, size_t const half_size_ceil_received,
    size_t const index_randomized_received,
    T *const ptr_array_shuffle_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  T const tmp_swap(ptr_array_shuffle_received[tmp_thread_global_index]);

  ptr_array_shuffle_received[tmp_thread_global_index] =
      ptr_array_shuffle_received[half_size_floor_received +
                                 ((index_randomized_received +
                                   tmp_thread_global_index) %
                                  half_size_ceil_received)];

  ptr_array_shuffle_received[half_size_floor_received +
                             ((index_randomized_received +
                               tmp_thread_global_index) %
                              half_size_ceil_received)] = tmp_swap;
}

template <typename T>
__global__ void kernel__Tree_Shift_Shuffle(
    size_t const size_received, size_t const half_size_floor_received,
    size_t const half_size_ceil_received,
    size_t const index_randomized_received,
    T *const ptr_array_shuffle_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  if (tmp_thread_global_index < size_received) {
    T const tmp_swap(ptr_array_shuffle_received[tmp_thread_global_index]);

    ptr_array_shuffle_received[tmp_thread_global_index] =
        ptr_array_shuffle_received[half_size_floor_received +
                                   ((index_randomized_received +
                                     tmp_thread_global_index) %
                                    half_size_ceil_received)];

    ptr_array_shuffle_received[half_size_floor_received +
                               ((index_randomized_received +
                                 tmp_thread_global_index) %
                                half_size_ceil_received)] = tmp_swap;
  }
}

template <typename T>
__global__ void kernel_while__Tree_Shift_Shuffle(
    size_t const size_received, size_t const half_size_floor_received,
    size_t const half_size_ceil_received,
    size_t const index_randomized_received,
    T *const ptr_array_shuffle_received) {
  size_t const tmp_grid_stride(gridDim.x * blockDim.x);
  size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  T tmp_swap;

  do {
    tmp_swap = ptr_array_shuffle_received[tmp_thread_global_index];

    ptr_array_shuffle_received[tmp_thread_global_index] =
        ptr_array_shuffle_received[half_size_floor_received +
                                   ((index_randomized_received +
                                     tmp_thread_global_index) %
                                    half_size_ceil_received)];

    ptr_array_shuffle_received[half_size_floor_received +
                               ((index_randomized_received +
                                 tmp_thread_global_index) %
                                half_size_ceil_received)] = tmp_swap;

    tmp_thread_global_index += tmp_grid_stride;
  } while (tmp_thread_global_index < size_received);
}

template <typename T>
__global__ void kernel_while__Tree_Shift_Shuffle_Power2(
    size_t const size_received, size_t const size_block_received,
    size_t const half_size_block_received,
    size_t const index_randomized_received,
    T *const ptr_array_shuffle_received) {
  size_t const tmp_grid_stride(gridDim.x * blockDim.x);
  size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x),
      tmp_tree_index(static_cast<size_t>(tmp_thread_global_index /
                                         half_size_block_received)),
      tmp_tree_thread_index(tmp_tree_index * half_size_block_received +
                            tmp_thread_global_index);

  T tmp_swap;

  while (tmp_tree_thread_index < size_received) {
    tmp_swap = ptr_array_shuffle_received[tmp_tree_thread_index];

    ptr_array_shuffle_received[tmp_tree_thread_index] =
        ptr_array_shuffle_received
            [tmp_tree_index * size_block_received + half_size_block_received +
             ((index_randomized_received + tmp_tree_thread_index) %
              half_size_block_received)];

    ptr_array_shuffle_received[tmp_tree_index * size_block_received +
                               half_size_block_received +
                               ((index_randomized_received +
                                 tmp_tree_thread_index) %
                                half_size_block_received)] = tmp_swap;

    tmp_thread_global_index += tmp_grid_stride;

    tmp_tree_index =
        static_cast<size_t>(tmp_thread_global_index / half_size_block_received);
    tmp_tree_thread_index =
        tmp_tree_index * half_size_block_received + tmp_thread_global_index;
  }
}

template <typename T>
__device__ inline void Shuffle_Loop(
    size_t const size_received, T *const ptr_array_shuffle_received,
    struct curandStateMtgp32 *const ptr_cuRAND_State_MTGP32_received) {
  size_t tmp_randomize_index, i;

  T tmp_swap;

  for (i = size_received; i--;) {
    tmp_randomize_index = static_cast<size_t>(
        curand(ptr_cuRAND_State_MTGP32_received) % (i + 1u));

    // Store the index to swap from the remaining index at "tmp_randomize_index"
    tmp_swap = ptr_array_shuffle_received[tmp_randomize_index];

    // Get remaining index starting at index "i"
    // And store it to the remaining index at "tmp_randomize_index"
    ptr_array_shuffle_received[tmp_randomize_index] =
        ptr_array_shuffle_received[i];

    // Store the swapped index at the index "i"
    ptr_array_shuffle_received[i] = tmp_swap;
  }
}

template <typename T>
__device__ void Tree_Shift_Shuffle(
    size_t const size_received, size_t const minimum_threads_occupancy_received,
    T *const ptr_array_shuffle_received,
    struct curandStateMtgp32 *const ptr_cuRAND_State_MTGP32_received,
    struct dim3 const *const ptr_dimension_grid_received,
    struct dim3 const *const ptr_dimension_block_received) {
  if (size_received > 1u) {
    if (USE_PARALLEL && size_received > minimum_threads_occupancy_received) {
      size_t tmp_shuffle_block_size(static_cast<size_t>(size_received / 2));

      /*
      INFO(L"size_received: %u" , size_received);
      INFO(L"size_half_floor: %u" , tmp_shuffle_block_size);
      INFO(L"size_half_ceil: %u" ,
      static_cast<size_t>(ceil(static_cast<double>(size_received) / 2.0)));
      INFO(L"grid(%u, %u, %u)" , ptr_dimension_grid_received->x,
      ptr_dimension_grid_received->y, ptr_dimension_grid_received->z);
      INFO(L"block(%u, %u, %u)" ,
      ptr_dimension_block_received->x, ptr_dimension_block_received->y,
      ptr_dimension_block_received->z);

      INFO(L"Before shuffle." );
      for(size_t i = 0u; i != size_received; ++i)
      { INFO(L"Index[%3u](%3u)" , i,
      ptr_array_shuffle_received[i]); }
      */

      LAUNCH_KERNEL_POINTER_1D(
          Tree_Shift_Shuffle<T>, ptr_dimension_grid_received,
          ptr_dimension_block_received, 0_UZ, tmp_shuffle_block_size,
          tmp_shuffle_block_size,
          static_cast<size_t>(ceil(static_cast<double>(size_received) / 2.0)),
          curand(ptr_cuRAND_State_MTGP32_received), ptr_array_shuffle_received);

      /*
      INFO(L"After shuffle." );
      for(size_t i = 0u; i != size_received; ++i)
      { INFO(L"Index[%3u](%3u)" , i,
      ptr_array_shuffle_received[i]); }

      CUDA__Check_Error();
      INFO(L"Check same..." );
      for(size_t i = 0, j; i != size_received; ++i)
      {
          for(j = i + 1u; j != size_received; ++j)
          {
              if(ptr_array_shuffle_received[i] == ptr_array_shuffle_received[j])
              { INFO(L"i[%3u](%3u) == j[%3u](%3u)" , i,
      ptr_array_shuffle_received[i], j, ptr_array_shuffle_received[j]); }
          }
      }

      INFO(L"Check present..." );
      for(size_t i = 0, j; i != size_received; ++i)
      {
          for(j = 0u; j != size_received; ++j)
          {
              if(i == ptr_array_shuffle_received[j])
              { break; }

              if(j + 1u == size_received)
              { INFO(L"i[%3u] Not present!" , i); }
          }
      }
      */

      if ((tmp_shuffle_block_size =
               DL::Math::Round_Down_At_Power_Of_Two<size_t>(
                   tmp_shuffle_block_size - 1u)) >=
          minimum_threads_occupancy_received) {
        size_t const tmp_shuffle_block_limit_size(tmp_shuffle_block_size * 2u);

        do {
          /*
          INFO(L"tmp_shuffle_block_limit_size: %u" ,
          tmp_shuffle_block_limit_size); INFO(L"tmp_shuffle_block_size *
          2: %u" , tmp_shuffle_block_size * 2u);
          INFO(L"tmp_shuffle_block_size: %u" ,
          tmp_shuffle_block_size); INFO(L"grid(%u, %u, %u)" ,
          ptr_dimension_grid_received[1].x, ptr_dimension_grid_received[1].y,
          ptr_dimension_grid_received[1].z); INFO(L"block(%u, %u, %u)"
          , ptr_dimension_block_received[1].x,
          ptr_dimension_block_received[1].y,
          ptr_dimension_block_received[1].z);

          INFO(L"Before shuffle." );
          for(size_t i = 0u; i != tmp_shuffle_block_limit_size; ++i)
          { INFO(L"Index[%3u](%3u)" , i,
          ptr_array_shuffle_received[i]); }
          */

          kernel_while__Tree_Shift_Shuffle_Power2<T>
              <<<ptr_dimension_grid_received[1],
                 ptr_dimension_block_received[1]>>>(
                  tmp_shuffle_block_limit_size, tmp_shuffle_block_size * 2,
                  tmp_shuffle_block_size,
                  curand(ptr_cuRAND_State_MTGP32_received),
                  ptr_array_shuffle_received);

          /*
          INFO(L"After shuffle." );
          for(size_t i = 0u; i != tmp_shuffle_block_limit_size; ++i)
          { INFO(L"Index[%3u](%3u)" , i,
          ptr_array_shuffle_received[i]); }

          CUDA__Check_Error();
          INFO(L"Check same..." );
          for(size_t i = 0, j; i != tmp_shuffle_block_limit_size; ++i)
          {
              for(j = i + 1u; j != tmp_shuffle_block_limit_size; ++j)
              {
                  if(ptr_array_shuffle_received[i] ==
          ptr_array_shuffle_received[j]) { INFO(L"i[%3u](%3u) ==
          j[%3u](%3u)" , i, ptr_array_shuffle_received[i], j,
          ptr_array_shuffle_received[j]); }
              }
          }

          INFO(L"Check present..." );
          for(size_t i = 0, j; i != tmp_shuffle_block_limit_size; ++i)
          {
              for(j = 0u; j != tmp_shuffle_block_limit_size; ++j)
              {
                  if(i == ptr_array_shuffle_received[j])
                  { break; }

                  if(j + 1u == tmp_shuffle_block_limit_size)
                  { INFO(L"i[%3u] Not present!" , i); }
              }
          }
          */
        } while ((tmp_shuffle_block_size =
                      DL::Math::Round_Down_At_Power_Of_Two<size_t>(
                          tmp_shuffle_block_size - 1u)) >=
                 minimum_threads_occupancy_received);
      }
    } else {
      Shuffle_Loop<T>(size_received, ptr_array_shuffle_received,
                      ptr_cuRAND_State_MTGP32_received);
    }
  } else if (size_received == 0u) {
    ERR(L"No array to shuffle!",);
  }
}
template __device__ void Tree_Shift_Shuffle(size_t const, size_t const,
                                            size_t *const,
                                            struct curandStateMtgp32 *const,
                                            struct dim3 const *const,
                                            struct dim3 const *const);

template <typename T>
__global__ void kernel__Tree_Shuffle(
    size_t const size_block_received, size_t const size_array_received,
    T *const ptr_array_shuffle_received,
    struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  T *const tmp_ptr_array_shuffle(ptr_array_shuffle_received +
                                 tmp_thread_global_index * size_block_received),
      tmp_swap;

  for (size_t tmp_randomize_index, i(size_block_received); i--;) {
    tmp_randomize_index = static_cast<size_t>(
        curand(ptr_array_cuRAND_State_MTGP32_received + blockIdx.x) % (i + 1u));

    if (tmp_thread_global_index * size_block_received + i <
        size_array_received) {
      // Store the index to swap from the remaining index at
      // "tmp_randomize_index"
      tmp_swap = tmp_ptr_array_shuffle[tmp_randomize_index];

      // Get remaining index starting at index "i"
      // And store it to the remaining index at "tmp_randomize_index"
      tmp_ptr_array_shuffle[tmp_randomize_index] = tmp_ptr_array_shuffle[i];

      // Store the swapped index at the index "i"
      tmp_ptr_array_shuffle[i] = tmp_swap;
    }
  }
}

template <typename T>
__global__ void kernel__Tree_Shuffle(
    size_t const size_received, size_t const size_block_received,
    size_t const size_array_received, T *const ptr_array_shuffle_received,
    struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received) {
  size_t const tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x);

  T *const tmp_ptr_array_shuffle(ptr_array_shuffle_received +
                                 tmp_thread_global_index * size_block_received),
      tmp_swap;

  for (size_t tmp_randomize_index, i(size_block_received); i--;) {
    tmp_randomize_index = static_cast<size_t>(
        curand(ptr_array_cuRAND_State_MTGP32_received + blockIdx.x) % (i + 1u));

    if (tmp_thread_global_index * size_block_received + i <
        size_array_received) {
      // Store the index to swap from the remaining index at
      // "tmp_randomize_index"
      tmp_swap = tmp_ptr_array_shuffle[tmp_randomize_index];

      // Get remaining index starting at index "i"
      // And store it to the remaining index at "tmp_randomize_index"
      tmp_ptr_array_shuffle[tmp_randomize_index] = tmp_ptr_array_shuffle[i];

      // Store the swapped index at the index "i"
      tmp_ptr_array_shuffle[i] = tmp_swap;
    }
  }
}

template <typename T>
__global__ void kernel_while__Tree_Shuffle(
    size_t const size_received, size_t const size_block_received,
    size_t const size_array_received, T *const ptr_array_shuffle_received,
    struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received) {
  size_t const tmp_grid_stride(gridDim.x * blockDim.x);
  size_t tmp_thread_global_index(blockIdx.x * blockDim.x + threadIdx.x),
      tmp_thread_block_index(blockIdx.x * blockDim.x);

  T *tmp_ptr_array_shuffle, tmp_swap;

  do {
    tmp_ptr_array_shuffle = ptr_array_shuffle_received +
                            tmp_thread_global_index * size_block_received;

    for (size_t tmp_randomize_index, i(size_block_received); i--;) {
      tmp_randomize_index = static_cast<size_t>(
          curand(ptr_array_cuRAND_State_MTGP32_received + blockIdx.x) %
          (i + 1u));

      if (tmp_thread_global_index * size_block_received + i <
          size_array_received) {
        // Store the index to swap from the remaining index at
        // "tmp_randomize_index"
        tmp_swap = tmp_ptr_array_shuffle[tmp_randomize_index];

        // Get remaining index starting at index "i"
        // And store it to the remaining index at "tmp_randomize_index"
        tmp_ptr_array_shuffle[tmp_randomize_index] = tmp_ptr_array_shuffle[i];

        // Store the swapped index at the index "i"
        tmp_ptr_array_shuffle[i] = tmp_swap;
      }
    }

    tmp_thread_global_index += tmp_grid_stride;
    tmp_thread_block_index += tmp_grid_stride;
  } while (tmp_thread_block_index < size_received);
}

template <typename T>
__device__ void Tree_Shuffle(
    size_t const size_received, size_t const size_block_received,
    size_t const size_array_received, T *const ptr_array_shuffle_received,
    struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received,
    struct dim3 const *const ptr_dimension_grid_received,
    struct dim3 const *const ptr_dimension_block_received) {
  if (size_array_received > 1u) {
    if (USE_PARALLEL && size_array_received >= size_block_received) {
      LAUNCH_KERNEL_POINTER_1D(Tree_Shuffle<T>, ptr_dimension_grid_received,
                               ptr_dimension_block_received, 0_UZ,
                               size_received, size_block_received,
                               size_array_received, ptr_array_shuffle_received,
                               ptr_array_cuRAND_State_MTGP32_received);
    } else {
      Shuffle_Loop<T>(size_array_received, ptr_array_shuffle_received,
                      ptr_array_cuRAND_State_MTGP32_received);
    }
  } else if (size_received == 0u) {
    ERR(L"No array to shuffle!",);
  }
}
template __device__ void Tree_Shuffle(size_t const, size_t const, size_t const,
                                      size_t *const,
                                      struct curandStateMtgp32 *const,
                                      struct dim3 const *const,
                                      struct dim3 const *const);
}  // namespace Shuffle
