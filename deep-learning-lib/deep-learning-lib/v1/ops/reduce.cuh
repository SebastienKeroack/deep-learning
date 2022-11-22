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

namespace DL::v1::Math {
template <typename T>
__device__ inline void Launch_Reduce(
    size_t const size_received, T *const ptr_array_outputs_received,
    T const *const ptr_array_inputs_received,
    struct dim3 const *const ptr_dimension_grid_recieved,
    struct dim3 const *const ptr_dimension_block_recieved);

template <typename T>
__device__ void Reduce(size_t const size_received,
                       size_t const stride_dim3_received,
                       T *const ptr_array_outputs_received,
                       T const *const ptr_array_inputs_received,
                       struct dim3 const *const ptr_dimension_grid_recieved,
                       struct dim3 const *const ptr_dimension_block_recieved);

template <typename T>
__device__ inline void Launch_Reduce_Square(
    size_t const size_received, T *const ptr_array_outputs_received,
    T const *const ptr_array_to_reduce_square_received,
    struct dim3 const *const ptr_dimension_grid_recieved,
    struct dim3 const *const ptr_dimension_block_recieved);

template <typename T>
__device__ void Reduce_Square(
    size_t const size_received, size_t const stride_dim3_received,
    T *const ptr_array_outputs_received,
    T const *const ptr_array_to_reduce_square_received,
    struct dim3 const *const ptr_dimension_grid_recieved,
    struct dim3 const *const ptr_dimension_block_recieved);

template <typename T>
__device__ inline void Launch_Reduce_XX(
    size_t const size_received, T *const ptr_array_outputs_received,
    T const *const ptr_array_X0_received, T const *const ptr_array_X1_received,
    struct dim3 const *const ptr_dimension_grid_recieved,
    struct dim3 const *const ptr_dimension_block_recieved);

template <typename T>
__device__ void Reduce_XX(
    size_t const size_received, size_t const stride_dim3_received,
    T *const ptr_array_outputs_received, T const *const ptr_array_X0_received,
    T const *const ptr_array_X1_received,
    struct dim3 const *const ptr_dimension_grid_recieved,
    struct dim3 const *const ptr_dimension_block_recieved);

template <typename T>
__device__ inline void Launch_Reduce_XZ(
    size_t const size_received, size_t const stride_Z_received,
    T *const ptr_array_outputs_received, T const *const ptr_array_X_received,
    T const *const ptr_array_Z_received,
    struct dim3 const *const ptr_dimension_grid_recieved,
    struct dim3 const *const ptr_dimension_block_recieved);

template <typename T>
__device__ void Reduce_XZ(
    size_t const size_received, size_t const stride_dim3_received,
    size_t const stride_Z_received, T *const ptr_array_outputs_received,
    T const *const ptr_array_X_received, T const *const ptr_array_Z_received,
    struct dim3 const *const ptr_dimension_grid_recieved,
    struct dim3 const *const ptr_dimension_block_recieved);

template <typename T>
__device__ inline void Launch_Reduce_Z0Z1(
    size_t const size_received, size_t const stride_Z0_received,
    size_t const stride_Z1_received, T *const ptr_array_outputs_received,
    T const *const ptr_array_Z0_received, T const *const ptr_array_Z1_received,
    struct dim3 const *const ptr_dimension_grid_recieved,
    struct dim3 const *const ptr_dimension_block_recieved);

template <typename T>
__device__ void Reduce_Z0Z1(
    size_t const size_received, size_t const stride_dim3_received,
    size_t const stride_Z0_received, size_t const stride_Z1_received,
    T *const ptr_array_outputs_received, T const *const ptr_array_Z0_received,
    T const *const ptr_array_Z1_received,
    struct dim3 const *const ptr_dimension_grid_recieved,
    struct dim3 const *const ptr_dimension_block_recieved);

template <typename T>
__device__ inline void Launch_Reduce_Array(
    size_t const size_received, size_t const stride_array_received,
    T *const ptr_array_IO_received,
    struct dim3 const *const ptr_dimension_grid_recieved,
    struct dim3 const *const ptr_dimension_block_recieved,
    struct dim3 const *const ptr_dimension_grid_reduce_array_recieved,
    struct dim3 const *const ptr_dimension_block_reduce_array_recieved);

template <typename T>
__device__ void Reduce_Array(
    size_t const size_received, size_t const stride_array_received,
    size_t const stride_dim3_received, T *const ptr_array_IO_received,
    struct dim3 const *const ptr_dimension_grid_recieved,
    struct dim3 const *const ptr_dimension_block_recieved,
    struct dim3 const *const ptr_dimension_grid_reduce_array_recieved,
    struct dim3 const *const ptr_dimension_block_reduce_array_recieved);
}  // namespace Reduce

#include "deep-learning-lib/ops/reduce.cu"