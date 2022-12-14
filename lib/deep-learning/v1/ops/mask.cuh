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

template <typename T>
__global__ void kernel__Flag_1D(bool const *const ptr_array_flag_received,
                                T *const ptr_array_to_one_received);

template <typename T>
__global__ void kernel__Flag_1D(size_t const size_received,
                                bool const *const ptr_array_flag_received,
                                T *const ptr_array_to_one_received);

template <typename T>
__global__ void kernel_while__Flag_1D(size_t const size_received,
                                      bool const *const ptr_array_flag_received,
                                      T *const ptr_array_to_one_received);

template <typename T>
__device__ void Flag_1D(size_t const size_received,
                        bool const *ptr_array_flag_received,
                        T *ptr_array_to_flag_received,
                        struct dim3 const *const ptr_dimension_grid_received,
                        struct dim3 const *const ptr_dimension_block_received);

#include "deep-learning/ops/mask.cu"