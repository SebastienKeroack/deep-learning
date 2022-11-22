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
__global__ void kernel__Multiply_Z_Y_1D(size_t const stride_Z_received,
                                        T const constant_received,
                                        T *const ptr_array_Z_received);

template <typename T>
__global__ void kernel__Multiply_Z_Y_1D(size_t const size_received,
                                        size_t const stride_Z_received,
                                        T const constant_received,
                                        T *const ptr_array_Z_received);

template <typename T>
__global__ void kernel_while__Multiply_Z_Y_1D(size_t const size_received,
                                              size_t const stride_Z_received,
                                              T const constant_received,
                                              T *const ptr_array_Z_received);

template <typename T>
__device__ void Multiply_Z_Y_1D(
    size_t const size_received, size_t const stride_Z_received,
    T const constant_received, T *ptr_array_Z_received,
    struct dim3 const *const ptr_dim3_grid_received,
    struct dim3 const *const ptr_dim3_block_received);

template <typename T>
__global__ void kernel__Multiply_X_Y_1D(T const constant_received,
                                        T *const ptr_array_X_received);

template <typename T>
__global__ void kernel__Multiply_X_Y_1D(size_t const size_received,
                                        T const constant_received,
                                        T *const ptr_array_X_received);

template <typename T>
__global__ void kernel_while__Multiply_X_Y_1D(size_t const size_received,
                                              T const constant_received,
                                              T *const ptr_array_X_received);

template <typename T>
__device__ void Multiply_X_Y_1D(
    size_t const size_received, T const constant_received,
    T *ptr_array_X_received, struct dim3 const *const ptr_dim3_grid_received,
    struct dim3 const *const ptr_dim3_block_received);

template <typename T>
__device__ void Multiply_X_Y_1D(
    bool &ref_synchronized_received, size_t const size_received,
    T const constant_received, T *ptr_array_X_received,
    struct dim3 const *const ptr_dim3_grid_received,
    struct dim3 const *const ptr_dim3_block_received);

template <typename T>
__global__ void kernel__FMAC_Z_YX_1D(size_t const stride_Z_received,
                                     T *const ptr_array_Z_received,
                                     T const constant_received,
                                     T const *const ptr_array_X_received);

template <typename T>
__global__ void kernel__FMAC_Z_YX_1D(size_t const size_received,
                                     size_t const stride_Z_received,
                                     T *const ptr_array_Z_received,
                                     T const constant_received,
                                     T const *const ptr_array_X_received);

template <typename T>
__global__ void kernel_while__FMAC_Z_YX_1D(size_t const size_received,
                                           size_t const stride_Z_received,
                                           T *const ptr_array_Z_received,
                                           T const constant_received,
                                           T const *const ptr_array_X_received);

template <typename T>
__device__ void FMAC_Z_YX_1D(
    size_t const size_received, size_t const stride_Z_received,
    T *ptr_array_Z_received, T const constant_received,
    T const *ptr_array_X_received,
    struct dim3 const *const ptr_dimension_grid_received,
    struct dim3 const *const ptr_dimension_block_received);

template <typename T>
__global__ void kernel__FMAC_X_YZ_1D(size_t const stride_Z_received,
                                     T *const ptr_array_X_received,
                                     T const constant_received,
                                     T const *const ptr_array_Z_received);

template <typename T>
__global__ void kernel__FMAC_X_YZ_1D(size_t const size_received,
                                     size_t const stride_Z_received,
                                     T *const ptr_array_X_received,
                                     T const constant_received,
                                     T const *const ptr_array_Z_received);

template <typename T>
__global__ void kernel_while__FMAC_X_YZ_1D(size_t const size_received,
                                           size_t const stride_Z_received,
                                           T *const ptr_array_X_received,
                                           T const constant_received,
                                           T const *const ptr_array_Z_received);

template <typename T>
__device__ void FMAC_X_YZ_1D(
    size_t const size_received, size_t const stride_Z_received,
    T *ptr_array_X_received, T const constant_received,
    T const *ptr_array_Z_received,
    struct dim3 const *const ptr_dimension_grid_received,
    struct dim3 const *const ptr_dimension_block_received);

template <typename T>
__global__ void kernel__FMAC_X_YX_1D(
    T *const ptr_array_outputs_X_received, T const constant_received,
    T const *const ptr_array_inputs_X_received);

template <typename T>
__global__ void kernel__FMAC_X_YX_1D(
    size_t const size_received, T *const ptr_array_outputs_X_received,
    T const constant_received, T const *const ptr_array_inputs_X_received);

template <typename T>
__global__ void kernel_while__FMAC_X_YX_1D(
    size_t const size_received, T *const ptr_array_outputs_X_received,
    T const constant_received, T const *const ptr_array_inputs_X_received);

template <typename T>
__device__ void FMAC_X_YX_1D(
    size_t const size_received, T *ptr_array_outputs_X_received,
    T const constant_received, T const *ptr_array_inputs_X_received,
    struct dim3 const *const ptr_dimension_grid_received,
    struct dim3 const *const ptr_dimension_block_received);

template <typename T>
__device__ void FMAC_X_YX_1D__atomic(
    size_t const size_received, T *ptr_array_outputs_X_received,
    T const constant_received, T const *ptr_array_inputs_X_received,
    struct dim3 const *const ptr_dimension_grid_received,
    struct dim3 const *const ptr_dimension_block_received);
}  // namespace Multiply

#include "deep-learning-lib/ops/multiply.cu"