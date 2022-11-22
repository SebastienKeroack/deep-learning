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
__device__ void Transpose_Square(
    size_t const size_received, size_t const width_received,
    T *const ptr_array_outputs_received,
    T const *const ptr_array_inputs_received,
    struct dim3 const *const ptr_dimension_grid_recieved,
    struct dim3 const *const ptr_dimension_block_recieved);

template <typename T>
__device__ void Transpose_Rectangular(
    size_t const size_received, size_t const columns_length_received,
    size_t const rows_length_received, T *const ptr_array_outputs_received,
    T const *const ptr_array_inputs_received,
    struct dim3 const *const ptr_dimension_grid_recieved,
    struct dim3 const *const ptr_dimension_block_recieved);

template <typename T>
__device__ void Transpose(
    size_t const size_received, size_t const columns_length_received,
    size_t const rows_length_received, T *const ptr_array_outputs_received,
    T const *const ptr_array_inputs_received,
    struct dim3 const *const ptr_dimension_grid_recieved,
    struct dim3 const *const ptr_dimension_block_recieved);
}  // namespace Transpose

#include "deep-learning-lib/ops/transpose.cu"