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

#include "pch.hpp"

#include "deep-learning/device/gpu/cuda/framework.cuh"
#include "deep-learning/device/gpu/cuda/info.cuh"

void CUDA__Print__Device_Property(struct cudaDeviceProp const &device,
                                  int const device_index) {
  // 1024 bytes = 1 KB, Kilobyte
  // 1024 KB = 1 MB, Megabyte
  // 1024 MB = 1 GB, Gibabyte
  // 1024 GB = 1 TB, Terabyte
  // 1024 TB = 1 PB, PB = Petabye
  // byte  to kilobye: 1024 bytes
  // byte  to megabye: 1024 bytes ^ 2 = 1 048 576 bytes
  // byte  to gigabyte: 1024 bytes ^ 3 = 1 073 741 824 bytes
  // 1000 Hertz = 1 kHz, kilohertz
  // 1000 kHz = 1 MHz, MegaHertz
  // 1000 MHz = 1 GHz, GigaHertz
  // 1000 GHz = 1 THz, TeraHertz
  // Hertz to kHz: 1000 Hertz
  // Hertz to MHz: 1000 ^ 2 = 1 000 000 Hertz
  // Hertz to GHz: 1000 ^ 3 = 1 000 000 000 Hertz
  if (device_index < 0) {
    return;
  }

  int main_index(0);
  CUDA__Safe_Call(cudaGetDevice(&main_index));

  size_t const n_cuda_cores(CUDA__Number_CUDA_Cores(device));
  size_t bytes_total(0), bytes_free(0);
  int driver_ver(0), runtime_ver(0);

  CUDA__Safe_Call(cudaSetDevice(device_index));

  INFO(L"Device [%d] name: \"%ls\"", device_index,
               device.name);

  CUDA__Safe_Call(cudaDriverGetVersion(&driver_ver));
  CUDA__Safe_Call(cudaRuntimeGetVersion(&runtime_ver));

  INFO(L"CUDA Driver version / Runtime version: %d / %d", driver_ver, runtime_ver);

  INFO(L"CUDA Capability Major/Minor version number: %d.%d", device.major,
               device.minor);

  INFO(
      L"Total amount of global memory: %.2f GB | %.2f MB | %zu "
      "bytes",
      static_cast<double>(device.totalGlobalMem) / 1073741824.0,
      static_cast<double>(device.totalGlobalMem) / 1048576.0,
      device.totalGlobalMem);

  CUDA__Safe_Call(cudaMemGetInfo(&bytes_free, &bytes_total));

  INFO(
      L"Total amount of free memory: %.2f GB | %.2f MB | %zu "
      "bytes",
      static_cast<double>(bytes_free) / 1073741824.0,
      static_cast<double>(bytes_free) / 1048576.0, bytes_free);
  INFO(
      L"Total amount of global available memory: %.2f GB | %.2f MB | %zu "
      "bytes",
      static_cast<double>(bytes_total) / 1073741824.0,
      static_cast<double>(bytes_total) / 1048576.0, bytes_total);

  INFO(
      L"(%d) Multiprocessors, (%zu) CUDA Cores/MP: %zu CUDA "
      "Cores", device.multiProcessorCount,
      n_cuda_cores,
      n_cuda_cores / static_cast<size_t>(device.multiProcessorCount));

  INFO(L"Clock frequency: %.4f GHz | %.4f MHz | %d kHz",
               static_cast<double>(device.clockRate) / 1000000.0,
               static_cast<double>(device.clockRate) / 1000.0,
               device.clockRate);

  INFO(
      L"Peak memory clock frequency: %.4f GHz | %.4f MHz | %d "
      "kHz",
      static_cast<double>(device.memoryClockRate) / 1000000.0,
      static_cast<double>(device.memoryClockRate) / 1000.0,
      device.memoryClockRate);

  INFO(L"Global memory bus width: %d-bit", device.memoryBusWidth);

  INFO(L"Size of L2 cache: %.2f MB | %.2f KB | %d bytes",
               static_cast<double>(device.l2CacheSize) / 1048576.0,
               static_cast<double>(device.l2CacheSize) / 1024.0,
               device.l2CacheSize);

  INFO(
      L"Maximum texture size (x, y, z): 1D=(%d), 2D=(%d, %d), 3D=(%d, %d, "
      "%d)", device.maxTexture1D,
      device.maxTexture2D[0], device.maxTexture2D[1],
      device.maxTexture3D[0], device.maxTexture3D[1],
      device.maxTexture3D[2]);

  INFO(
      L"Maximum mipmapped texture size (x, y): 1D=(%d), 2D=(%d, "
      "%d)", device.maxTexture1DMipmap,
      device.maxTexture2DMipmap[0], device.maxTexture2DMipmap[1]);

  INFO(
      L"Maximum textures bound to linear memory (x, y, pitch): 1D=(%d), "
      "2D=(%d, %d, %d)", device.maxTexture1DLinear,
      device.maxTexture2DLinear[0], device.maxTexture2DLinear[1],
      device.maxTexture2DLinear[2]);

  INFO(
      L"Maximum 2D texture dimensions if texture gather operations have "
      "to be performed (x, y): (%d, %d)", device.maxTexture2DGather[0],
      device.maxTexture2DGather[1]);

  INFO(
      L"Maximum alternate 3D texture dimensions (x, y, z): (%d, %d, "
      "%d)", device.maxTexture3DAlt[0],
      device.maxTexture3DAlt[1], device.maxTexture3DAlt[2]);

  INFO(L"Maximum Cubemap texture dimensions: %d",
               device.maxTextureCubemap);

  INFO(
      L"Maximum Cubemap texture dimensions: 1D=(%d), %d layers",
      device.maxTextureCubemapLayered[0],
      device.maxTextureCubemapLayered[1]);

  INFO(
      L"Maximum Cubemap layered texture dimensions, (num) layers: "
      "1D=(%d), %d layers", device.maxTexture1DLayered[0],
      device.maxTexture1DLayered[1]);

  INFO(
      L"Maximum 2D layered texture dimensions, (num) layers: 2D=(%d, %d), "
      "%d layers", device.maxTexture2DLayered[0],
      device.maxTexture2DLayered[1], device.maxTexture2DLayered[2]);

  INFO(
      L"Maximum surface size (x, y, z): 1D=(%d), 2D=(%d, %d), 3D=(%d, %d, "
      "%d)", device.maxSurface1D,
      device.maxSurface2D[0], device.maxSurface2D[1],
      device.maxSurface3D[0], device.maxSurface3D[1],
      device.maxSurface3D[2]);

  INFO(
      L"Maximum 1D layered surface dimensions, (num) layers: 1D=(%d), %d "
      "layers", device.maxSurface1DLayered[0],
      device.maxSurface1DLayered[1]);

  INFO(
      L"Maximum 2D layered surface dimensions, (num) layers: 2D=(%d, %d), "
      "%d layers", device.maxSurface2DLayered[0],
      device.maxSurface2DLayered[1], device.maxSurface2DLayered[2]);

  INFO(L"Maximum Cubemap surface dimensions: %d",
               device.maxSurfaceCubemap);

  INFO(
      L"Maximum Cubemap surface dimensions: 1D=(%d), %d layers",
      device.maxSurfaceCubemapLayered[0],
      device.maxSurfaceCubemapLayered[1]);

  INFO(
      L"Constant memory available on device: %.2f MB | %.2f KB | %zu "
      "bytes",
      static_cast<double>(device.totalConstMem) / 1048576.0,
      static_cast<double>(device.totalConstMem) / 1024.0,
      device.totalConstMem);

  INFO(
      L"Shared memory available per block: %.2f MB | %.2f KB | %zu "
      "bytes",
      static_cast<double>(device.sharedMemPerBlock) / 1048576.0,
      static_cast<double>(device.sharedMemPerBlock) / 1024.0,
      device.sharedMemPerBlock);

  INFO(L"32-bit registers available per block: %d", device.regsPerBlock);

  INFO(L"32-bit registers available per multiprocessor: %d",
               device.regsPerMultiprocessor);

  INFO(L"Warp size: %d",
               device.warpSize);

  INFO(L"Maximum number of threads per multiprocessor: %d",
               device.maxThreadsPerMultiProcessor);

  INFO(L"Maximum number of threads per block: %d",
               device.maxThreadsPerBlock);

  INFO(
      L"Maximum size of each dimension of a block (x, y, z): (%d, %d, "
      "%d)", device.maxThreadsDim[0],
      device.maxThreadsDim[1], device.maxThreadsDim[2]);

  INFO(
      L"Maximum size of each dimension of a grid (x, y, z): (%d, %d, "
      "%d)", device.maxGridSize[0],
      device.maxGridSize[1], device.maxGridSize[2]);

  INFO(
      L"Maximum pitch in bytes allowed by memory copies: %.2f MB | %.2f "
      "KB | %zu bytes",
      static_cast<double>(device.memPitch) / 1048576.0,
      static_cast<double>(device.memPitch) / 1024.0, device.memPitch);

  INFO(
      L"Alignment requirement for textures: %.2f MB | %.2f KB | %zu "
      "bytes",
      static_cast<double>(device.textureAlignment) / 1048576.0,
      static_cast<double>(device.textureAlignment) / 1024.0,
      device.textureAlignment);

  INFO(
      L"Pitch alignment requirement for texture references bound to "
      "pitched memory: %.2f MB | %.2f KB | %zu bytes",
      static_cast<double>(device.texturePitchAlignment) / 1048576.0,
      static_cast<double>(device.texturePitchAlignment) / 1024.0,
      device.texturePitchAlignment);

  INFO(
      L"Shared memory available per multiprocessor: %.2f MB | %.2f KB | "
      "%zu bytes",
      static_cast<double>(device.sharedMemPerMultiprocessor) / 1048576.0,
      static_cast<double>(device.sharedMemPerMultiprocessor) / 1024.0,
      device.sharedMemPerMultiprocessor);

  INFO(
      L"Per device maximum shared memory per block usable by special opt "
      "in: %.2f MB | %.2f KB | %zu bytes",
      static_cast<double>(device.sharedMemPerBlockOptin) / 1048576.0,
      static_cast<double>(device.sharedMemPerBlockOptin) / 1024.0,
      device.sharedMemPerBlockOptin);

  INFO(L"Number of asynchronous engines: %d",
               device.asyncEngineCount);

  INFO(
      L"Ratio of single precision performance (in floating-point "
      "operations per second) to double precision performance: %d",
      device.singleToDoublePrecisionPerfRatio);

  INFO(L"Alignment requirement for surfaces: %ls (%zu)",
               device.surfaceAlignment != 0_UZ ? "Yes" : "No",
               device.surfaceAlignment);

  INFO(
      L"Device can coherently access managed memory concurrently with the "
      "CPU: %ls",
      device.concurrentManagedAccess != 0 ? "Yes" : "No");

  INFO(
      L"Device can possibly execute multiple kernels concurrently: "
      "%ls",
      device.concurrentKernels != 0 ? "Yes" : "No");

  INFO(
      L"Device can access host registered memory at the same virtual "
      "address as the CPU: %ls",
      device.canUseHostPointerForRegisteredMem != 0 ? "Yes" : "No");

  INFO(
      L"Link between the device and the host supports native atomic "
      "operations: %ls",
      device.hostNativeAtomicSupported != 0 ? "Yes" : "No");

  INFO(L"Device is on a multi-GPU board: %ls",
               device.isMultiGpuBoard != 0 ? "Yes" : "No");

  INFO(
      L"Unique identifier for a group of devices on the same multi-GPU "
      "board: %d", device.multiGpuBoardGroupID);

  INFO(
      L"Device can map host memory with "
      "cudaHostAlloc/cudaHostGetDevicePointer: %ls",
      device.canMapHostMemory != 0 ? "Yes" : "No");

  INFO(L"Device supports stream priorities: %ls",
               device.streamPrioritiesSupported != 0 ? "Yes" : "No");

  INFO(L"Device supports caching globals in L1: %ls",
               device.globalL1CacheSupported != 0 ? "Yes" : "No");

  INFO(L"Device supports caching locals in L1: %ls",
               device.localL1CacheSupported != 0 ? "Yes" : "No");

  INFO(L"Device supports Compute Preemption: %ls",
               device.computePreemptionSupported != 0 ? "Yes" : "No");

  INFO(
      L"Device supports allocating managed memory on this system: "
      "%ls",
      device.managedMemory != 0 ? "Yes" : "No");

  INFO(
      L"Device supports launching cooperative kernels via "
      "::cudaLaunchCooperativeKernel: %ls",
      device.cooperativeLaunch != 0 ? "Yes" : "No");

  INFO(
      L"Device can participate in cooperative kernels launched via "
      "::cudaLaunchCooperativeKernelMultiDevice: %ls",
      device.cooperativeMultiDeviceLaunch != 0 ? "Yes" : "No");

  if (device.deviceOverlap != 0)  // [Deprecated]
  {
    INFO(
        L"Concurrent copy and execution [Deprecated]: Yes (maximum of "
        "%d)",
        CUDA__Maximum_Concurrent_Kernel(device));
  } else {
    INFO(L"Concurrent copy and execution [Deprecated]: No");
  }

  INFO(
      L"Specified whether there is a run time limit on kernels: "
      "%ls",
      device.kernelExecTimeoutEnabled != 0 ? "Yes" : "No");

  INFO(L"Device is integrated as opposed to discrete: %ls",
               device.integrated != 0 ? "Yes" : "No");

  INFO(
      L"Device supports coherently accessing pageable memory without "
      "calling cudaHostRegister on it: %ls",
      device.pageableMemoryAccess != 0 ? "Yes" : "No");

  INFO(L"Device has ECC support: %ls",
               device.ECCEnabled != 0 ? "Enabled" : "Disabled");

  INFO(L"CUDA Device Driver Mode (TCC or WDDM): %ls",
               device.tccDriver != 0
                   ? "TCC"
                   : "WDDM (Windows Display Driver Model)");

  INFO(
      L"Device shares a unified address space with the host (UVA): "
      "%ls",
      device.unifiedAddressing != 0 ? "Yes" : "No");

  INFO(L"PCI bus ID of the device: %d", device.pciBusID);

  INFO(L"PCI device ID of the device: %d", device.pciDeviceID);

  INFO(L"PCI domain ID of the device: %d", device.pciDomainID);

  INFO(L"Compute mode: [%d]", device.computeMode);

  switch (device.computeMode) {
    case 0:
      INFO(L"Default compute mode (Multiple threads can use cudaSetDevice() "
          "with this device).");
      break;
    case 1:
      INFO(L"Compute-exclusive-thread mode (Only one thread in one process "
          "will be able to use cudaSetDevice() with this device).");
      break;
    case 2:
      INFO(L"Compute-prohibited mode (No threads can use cudaSetDevice() with "
          "this device).");
      break;
    case 3:
      INFO(L"Compute-exclusive-process mode (Many threads in one process will "
          "be able to use cudaSetDevice() with this device).");
      break;
  }

  // https://devblogs.nvidia.com/parallelforall/how-implement-performance-metrics-cuda-cc/
  // (MHz * ((BusWidth to bytes) * double data rate) / Convert to GB/s)
  INFO(
      L"Theoretical bandwidth (GB/s): %f.",
      ((static_cast<double>(device.memoryClockRate) * 1000.0) *
       ((static_cast<double>(device.memoryBusWidth) / 8.0) * 2.0)) /
          1e9);

  // CudaCores * SMs * ClockRate_GHz * FMA_Instruction(2)
  INFO(
      L"Single precision: Theoretical throughput (GFLOP/s): %f.",
      static_cast<double>(CUDA__Number_CUDA_Cores(device)) *
          (static_cast<double>(device.clockRate) / 1e6));
  INFO(
      L"Single precision: Theoretical throughput FMA (GFLOP/s): "
      "%f.",
      static_cast<double>(CUDA__Number_CUDA_Cores(device)) *
          (static_cast<double>(device.clockRate) / 1e6) * 2.0);
  // FMA = 1 instruction for 2 flops, Fused multiply-add (X * Y + Z)

  if (device_index != main_index) {
    CUDA__Safe_Call(cudaSetDevice(main_index));
  }
}

int CUDA__Device_Count(void) {
  int n_devices(0);
  CUDA__Safe_Call(cudaGetDeviceCount(&n_devices));
  return n_devices;
}

int CUDA__Maximum_Concurrent_Kernel(struct cudaDeviceProp const &device) {
  unsigned int max_concurrent_kernel(0u);

  switch (device.major) {
    case 2:  // Fermi.
      max_concurrent_kernel = 16u;
      break;
    case 3:  // Kepler.
      if (device.minor == 2) {
        max_concurrent_kernel = 4u;
      }
      if (device.minor == 5 || device.minor == 7) {
        max_concurrent_kernel = 32u;
      } else {
        INFO(L"Unknown minor device version.");
      }
      break;
    case 5:  // Maxwell.
      if (device.minor == 0 || device.minor == 2) {
        max_concurrent_kernel = 32u;
      } else if (device.minor == 3) {
        max_concurrent_kernel = 16u;
      } else {
        INFO(L"Unknown minor device version.");
      }
      break;
    case 6:  // Pascal.
      if (device.minor == 0) {
        max_concurrent_kernel = 128u;
      } else if (device.minor == 1) {
        max_concurrent_kernel = 32u;
      } else if (device.minor == 2) {
        max_concurrent_kernel = 16u;
      } else {
        INFO(L"Unknown minor device version.");
      }
      break;
    case 7:  // Volta.
      if (device.minor == 0) {
        max_concurrent_kernel = 128u;
      } else {
        INFO(L"Unknown minor device version.");
      }
      break;
    default:
      INFO(L"Unknown major device version.");
      break;
  }

  return max_concurrent_kernel;
}

size_t CUDA__Number_CUDA_Cores(struct cudaDeviceProp const &device) {
  size_t const n_multiprocessor(
      static_cast<size_t>(device.multiProcessorCount));
  size_t n_cores(0);

  switch (device.major) {
    case 2:  // Fermi.
      if (device.minor == 1) {
        n_cores = n_multiprocessor * 48_UZ;
      } else {
        n_cores = n_multiprocessor * 32_UZ;
      }
      break;
    case 3:  // Kepler.
             // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-3-0
      n_cores = n_multiprocessor * 192_UZ;
      break;
    case 5:  // Maxwell.
             // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-5-x
      n_cores = n_multiprocessor * 128_UZ;
      break;
    case 6:  // Pascal.
             // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-6-x
      if (device.minor == 0) {
        n_cores = n_multiprocessor * 64_UZ;
      } else if (device.minor == 1 || device.minor == 2) {
        n_cores = n_multiprocessor * 128_UZ;
      } else {
        INFO(L"Unknown minor device version.");
      }
      break;
    case 7:  // Volta.
             // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-7-x
      if (device.minor == 0) {
        n_cores = n_multiprocessor * 64_UZ;
      } else {
        INFO(L"Unknown minor device version.");
      }
      break;
    default:
      INFO(L"Unknown major device version.");
      break;
  }

  return n_cores;
}

bool CUDA__Input__Use__CUDA(int &device_index,
                            size_t &bytes_maximum_allowable) {
  unsigned int const n_devices(CUDA__Device_Count());

  if (n_devices == 0u) {
    return false;
  }

  bool enabled(false);

  struct cudaDeviceProp device;

  for (int i(0); i != n_devices; ++i) {
    CUDA__Safe_Call(cudaGetDeviceProperties(&device, static_cast<int>(i)));

    CUDA__Print__Device_Property(device, i);

    if (CUDA__Required_Compatibility_Device(device)) {
      enabled =
          DL::Term::accept(DL::Str::now_format() +
                               ": Do you want to enable CUDA with this card: " +
                               std::string(device.name) + "?");

      device_index = i;

      break;
    }
  }

  if (enabled) {
    CUDA__Set__Device(device_index);

    CUDA__Safe_Call(cudaGetDeviceProperties(&device, device_index));

    INFO(L"");
    INFO(L"Device[%d]: %ls.", device_index,
                 device.name);

    size_t bytes_total(0), bytes_free(0);

    CUDA__Safe_Call(cudaMemGetInfo(&bytes_free, &bytes_total));

    if (bytes_free / KILOBYTE / KILOBYTE < 1_UZ) {
      ERR(
          L"Not enough memory to use the GPU %zu "
          "bytes", bytes_free);

      return false;
    }

    INFO(L"");
    INFO(L"Maximum allowable memory.");
    INFO(L"Range[1, %zu] MB(s).",
                 bytes_free / KILOBYTE / KILOBYTE);

    bytes_maximum_allowable = DL::Term::parse_discrete(
        1_UZ, bytes_free / KILOBYTE / KILOBYTE,
        DL::Str::now_format() + ": Maximum allowable memory (MBs):");

    CUDA__Initialize__Device(device,
                             bytes_maximum_allowable * KILOBYTE * KILOBYTE);

    CUDA__Set__Synchronization_Depth(3_UZ);
  }

  return enabled;
}
