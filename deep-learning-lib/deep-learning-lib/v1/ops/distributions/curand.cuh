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

__global__ void kernel__cuRAND__Memcpy_cuRAND_State_MTGP32(
    struct curandStateMtgp32
        *const ptr_array_cuRAND_State_MTGP32_destination_received,
    struct curandStateMtgp32
        *const ptr_array_cuRAND_State_MTGP32_source_received,
    struct mtgp32_kernel_params
        *const ptr_array_mtgp32_kernel_params_t_source_received);

__global__ void kernel__cuRAND__Memcpy_cuRAND_State_MTGP32(
    size_t const size_received,
    struct curandStateMtgp32
        *const ptr_array_cuRAND_State_MTGP32_destination_received,
    struct curandStateMtgp32
        *const ptr_array_cuRAND_State_MTGP32_source_received,
    struct mtgp32_kernel_params
        *const ptr_array_mtgp32_kernel_params_t_source_received);

__global__ void kernel_while__cuRAND__Memcpy_cuRAND_State_MTGP32(
    size_t const size_received,
    struct curandStateMtgp32
        *const ptr_array_cuRAND_State_MTGP32_destination_received,
    struct curandStateMtgp32
        *const ptr_array_cuRAND_State_MTGP32_source_received,
    struct mtgp32_kernel_params
        *const ptr_array_mtgp32_kernel_params_t_source_received);

__device__ void cuRAND__Memcpy_cuRAND_State_MTGP32(
    size_t const size_received,
    struct curandStateMtgp32
        *const ptr_array_cuRAND_State_MTGP32_destination_received,
    struct curandStateMtgp32
        *const ptr_array_cuRAND_State_MTGP32_source_received,
    struct mtgp32_kernel_params
        *const ptr_array_mtgp32_kernel_params_t_source_received,
    struct dim3 const *const ptr_dimension_grid_received,
    struct dim3 const *const ptr_dimension_block_received);

__host__ bool Allocate_cuRAND_MTGP32(
    int const number_states_MTGP32_received, size_t seed,
    struct mtgp32_kernel_params *&ptr_mtgp32_kernel_params_received,
    struct curandStateMtgp32 *&ptr_curandStateMtgp32_t_received);

__host__ void Cleanup_cuRAND_MTGP32(
    struct mtgp32_kernel_params *&ptr_mtgp32_kernel_params_received,
    struct curandStateMtgp32 *&ptr_curandStateMtgp32_t_received);