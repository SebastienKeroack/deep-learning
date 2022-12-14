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

#include <device_launch_parameters.h>

namespace DL::v1::Mem {
// TODO: CUDA Parallelism.
template <class T, bool STD = true>
__host__ __device__ void copy(T const *ptr_array_source_received,
                              T const *ptr_array_last_source_received,
                              T *ptr_array_destination_received);

// TODO: CUDA Parallelism.
template <class T, bool STD = true>
__host__ __device__ void fill(T *ptr_array_received,
                              T *const ptr_array_last_received,
                              T const value_received);

template <class T>
__host__ __device__ void fill_null(T *ptr_array_received,
                                      T const *const ptr_array_last_received);

template <class T, bool CPY = true, bool SET = true>
__host__ __device__ T *reallocate(T *ptr_array_received,
                                  size_t const new_size_received,
                                  size_t const old_size_received);

// TODO: WARNING dimension set/cpy are inverted.
template <class T, bool CPY = true, bool SET = true>
__device__ T *reallocate(T *ptr_array_received, size_t const new_size_received,
                         size_t const old_size_received,
                         struct dim3 const &ref_dimension_grid_set_received,
                         struct dim3 const &ref_dimension_block_set_received,
                         struct dim3 const &ref_dimension_grid_cpy_received,
                         struct dim3 const &ref_dimension_block_cpy_received);

template <class T, bool CPY = true>
__host__ __device__ T *reallocate_obj(T *ptr_array_received,
                                          size_t const new_size_received,
                                          size_t const old_size_received);

// TODO: WARNING dimension set/cpy are inverted.
template <class T, bool CPY = true>
__device__ T *reallocate_obj(
    T *ptr_array_received, size_t const new_size_received,
    size_t const old_size_received,
    struct dim3 const &ref_dimension_grid_set_received,
    struct dim3 const &ref_dimension_block_set_received,
    struct dim3 const &ref_dimension_grid_cpy_received,
    struct dim3 const &ref_dimension_block_cpy_received);

template <class T, bool CPY = true, bool SET = true>
__host__ __device__ T *reallocate_ptofpt(T *ptr_array_received,
                                         size_t const new_size_received,
                                         size_t const old_size_received);

// TODO: WARNING dimension set/cpy are inverted.
template <class T, bool CPY = true, bool SET = true>
__device__ T *reallocate_ptofpt(
    T *ptr_array_received, size_t const new_size_received,
    size_t const old_size_received,
    struct dim3 const &ref_dimension_grid_set_received,
    struct dim3 const &ref_dimension_block_set_received,
    struct dim3 const &ref_dimension_grid_cpy_received,
    struct dim3 const &ref_dimension_block_cpy_received);
} // namespace DL::Memory

#include "deep-learning/mem/reallocate.cu"