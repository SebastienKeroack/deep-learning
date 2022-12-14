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

#include <cuda_runtime.h>

__host__ __device__ static bool CUDA__Required_Compatibility_Device(
    struct cudaDeviceProp const &device) {
  return (device.major == 3 && device.minor >= 5) || device.major >= 4;
}

void CUDA__Print__Device_Property(struct cudaDeviceProp const &device,
                                  int const device_index);

int CUDA__Device_Count(void);

int CUDA__Maximum_Concurrent_Kernel(struct cudaDeviceProp const &device);

size_t CUDA__Number_CUDA_Cores(struct cudaDeviceProp const &device);

bool CUDA__Input__Use__CUDA(int &device_index, size_t &bytes_maximum_allowable);