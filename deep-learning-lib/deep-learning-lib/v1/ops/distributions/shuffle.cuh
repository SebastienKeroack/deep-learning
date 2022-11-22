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

struct curandStateMtgp32;

namespace DL::v1::Dist {
template <typename T>
__device__ void Tree_Shift_Shuffle(
    size_t const size_received, size_t const minimum_threads_occupancy_received,
    T *const ptr_array_shuffle_received,
    struct curandStateMtgp32 *const ptr_cuRAND_State_MTGP32_received,
    struct dim3 const *const ptr_dimension_grid_received,
    struct dim3 const *const ptr_dimension_block_received);

template <typename T>
__device__ void Tree_Shuffle(
    size_t const size_received, size_t const size_block_received,
    size_t const size_array_received, T *const ptr_array_shuffle_received,
    struct curandStateMtgp32 *const ptr_array_cuRAND_State_MTGP32_received,
    struct dim3 const *const ptr_dimension_grid_received,
    struct dim3 const *const ptr_dimension_block_received);
}  // namespace Shuffle