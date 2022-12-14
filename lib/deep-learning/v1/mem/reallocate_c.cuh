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

namespace DL::v1::Mem::C {
// TODO: CUDA Parallelism.
template <class T, bool CPY = true, bool SET = true>
__host__ __device__ T* reallocate(T* ptr_array_received,
                                  size_t const new_size_t_received,
                                  size_t const old_size_t_received);
}  // namespace DL::Memory::C

#include "deep-learning/mem/reallocate_c.cu"