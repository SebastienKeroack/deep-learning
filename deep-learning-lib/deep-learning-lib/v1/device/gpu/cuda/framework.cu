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

#include "deep-learning-lib/device/gpu/cuda/framework.cuh"

void CUDA__Initialize__Device(struct cudaDeviceProp const &device_prop,
                              size_t const memory_allocate) {
  size_t bytes_total(0), bytes_free(0);

  CUDA__Safe_Call(cudaMemGetInfo(&bytes_free, &bytes_total));

  if (bytes_free < memory_allocate) {
    CUDA__Safe_Call(
        cudaDeviceSetLimit(cudaLimit::cudaLimitMallocHeapSize, bytes_free));
  } else {
    CUDA__Safe_Call(cudaDeviceSetLimit(cudaLimit::cudaLimitMallocHeapSize,
                                       memory_allocate));
  }
}

void CUDA__Set__Device(int const device_index) {
  if (device_index >= 0) {
    CUDA__Safe_Call(cudaSetDevice(device_index));
  } else {
    ERR(L"Device index can not be less than one.",);
  }
}

void CUDA__Set__Synchronization_Depth(size_t const depth) {
  CUDA__Safe_Call(
      cudaDeviceSetLimit(cudaLimit::cudaLimitDevRuntimeSyncDepth, depth));
}

void CUDA__Reset(void) { CUDA__Safe_Call(cudaDeviceReset()); }
