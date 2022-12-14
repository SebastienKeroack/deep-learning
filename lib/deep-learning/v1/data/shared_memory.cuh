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

#define PREPROCESSED_CONCAT_(x, y) x##y
#define PREPROCESSED_CONCAT(x, y) PREPROCESSED_CONCAT_(x, y)

#define DECLARE_EXTERN_SHARED_MEMORY_TEMPLATE(struct_name_received)            \
  template <typename T>                                                        \
  struct struct_name_received {                                                \
    __device__ T* operator()(void) {                                           \
      extern __shared__ T PREPROCESSED_CONCAT(tmp_ptr_pointer_T_,              \
                                              struct_name_received)[];         \
      return (PREPROCESSED_CONCAT(tmp_ptr_pointer_T_, struct_name_received));  \
    }                                                                          \
  };                                                                           \
  template <>                                                                  \
  struct struct_name_received<int> {                                           \
    __device__ int* operator()(void) {                                         \
      extern __shared__ int PREPROCESSED_CONCAT(tmp_ptr_pointer_int_,          \
                                                struct_name_received)[];       \
      return (                                                                 \
          PREPROCESSED_CONCAT(tmp_ptr_pointer_int_, struct_name_received));    \
    }                                                                          \
  };                                                                           \
  template <>                                                                  \
  struct struct_name_received<unsigned int> {                                  \
    __device__ unsigned int* operator()(void) {                                \
      extern __shared__ unsigned int PREPROCESSED_CONCAT(                      \
          tmp_ptr_pointer_unsigned_int_, struct_name_received)[];              \
      return (PREPROCESSED_CONCAT(tmp_ptr_pointer_unsigned_int_,               \
                                  struct_name_received));                      \
    }                                                                          \
  };                                                                           \
  template <>                                                                  \
  struct struct_name_received<float> {                                         \
    __device__ float* operator()(void) {                                       \
      extern __shared__ float PREPROCESSED_CONCAT(tmp_ptr_pointer_float_,      \
                                                  struct_name_received)[];     \
      return (                                                                 \
          PREPROCESSED_CONCAT(tmp_ptr_pointer_float_, struct_name_received));  \
    }                                                                          \
  };                                                                           \
  template <>                                                                  \
  struct struct_name_received<double> {                                        \
    __device__ double* operator()(void) {                                      \
      extern __shared__ double PREPROCESSED_CONCAT(tmp_ptr_pointer_double_,    \
                                                   struct_name_received)[];    \
      return (                                                                 \
          PREPROCESSED_CONCAT(tmp_ptr_pointer_double_, struct_name_received)); \
    }                                                                          \
  };

#define EXTERN_SHARED_MEMORY_TEMPLATE(type_received, assignation_received, \
                                      struct_name_received)                \
  struct struct_name_received<type_received> tmp_struct_shared_memory;     \
  assignation_received = tmp_struct_shared_memory();
