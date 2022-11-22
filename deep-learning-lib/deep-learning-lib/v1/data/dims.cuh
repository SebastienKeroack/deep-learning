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

enum ENUM_TYPE_DIM3 : unsigned int {
  TYPE_DIM3_1D = 0,
  TYPE_DIM3_DYNAMIC_PARALLELISM = 1u
};

class cuDims {
 public:
  __device__ cuDims(void);
  __device__ ~cuDims(void);

  __device__ bool Get__Dim3(
      size_t const size_need_received, struct dim3 &ref_dim3_grid_received,
      struct dim3 &ref_dim3_block_received,
      class cuDeviceProp const *const ptr_Class_Device_Information_received,
      enum ENUM_TYPE_DIM3 const type_dim3_received);
  __device__ bool Get__Dim3_1D(
      size_t const size_need_received, struct dim3 &ref_dim3_grid_received,
      struct dim3 &ref_dim3_block_received,
      class cuDeviceProp const *const ptr_Class_Device_Information_received);
  __device__ bool Get__Dim3_Memcpy(
      size_t const new_size_received, size_t const old_size_received,
      struct dim3 &ref_dim3_grid_zero_received,
      struct dim3 &ref_dim3_block_zero_received,
      struct dim3 &ref_dim3_grid_copy_received,
      struct dim3 &ref_dim3_block_copy_received,
      class cuDeviceProp const *const ptr_Class_Device_Information_received,
      bool const memcpy_received = true);
  __device__ bool Get__Dim3_Dynamic_Parallelisme(
      size_t const size_need_received, struct dim3 &ref_dim3_grid_received,
      struct dim3 &ref_dim3_block_received,
      class cuDeviceProp const *const ptr_Class_Device_Information_received);

 private:
  int _size_1D = 0;
  int _size_DP = 0;

  size_t *_ptr_array_cache_dim3_size_1D = nullptr;
  size_t *_ptr_array_cache_dim3_size_DP = nullptr;

  struct dim3 *_ptr_array_dim3_grids_1D = NULL;
  struct dim3 *_ptr_array_dim3_blocks_1D = NULL;

  struct dim3 *_ptr_array_dim3_grids_DP = NULL;
  struct dim3 *_ptr_array_dim3_blocks_DP = NULL;
};
