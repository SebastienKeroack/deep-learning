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

#include "deep-learning/v1/ops/distributions/shuffle.cuh"
#include "deep-learning/v1/ops/distributions/curand.cuh"
#include "deep-learning/v1/learner/model.hpp"
#include "deep-learning/v1/learner/model.cuh"
#include "deep-learning/data/dataset.cuh"
#include "deep-learning/io/file.hpp"

#include <curand_kernel.h>

template <typename T>
__host__ __device__ cuDataset<T>::cuDataset(void) {}

template <typename T>
__global__ void kernel__Dataset_device__Add_CUDA_Device(
    int const index_device_received,
    struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received,
    class cuDataset<T> *const ptr_Dataset_device_received) {
  ptr_Dataset_device_received->Add_CUDA_Device(
      index_device_received, ptr_struct_cudaDeviceProp_received);
}

template <typename T>
__device__ bool cuDataset<T>::Add_CUDA_Device(
    int const index_device_received,
    struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received) {
  if (this->p_ptr_Class_Device_Information_Array == nullptr) {
    this->p_ptr_Class_Device_Information_Array = new class cuDevicesProp;
  }

  return (this->p_ptr_Class_Device_Information_Array->push_back(
      index_device_received, ptr_struct_cudaDeviceProp_received));
}

template <typename T>
__host__ bool cuDataset<T>::Initialize_CUDA_Device(void) {
  int device_id(0), tmp_number_CUDA_devices;

  struct cudaDeviceProp tmp_struct_cudaDeviceProp,
      *tmp_ptr_device_struct_cudaDeviceProp(NULL);

  CUDA__Safe_Call(cudaGetDeviceCount(&tmp_number_CUDA_devices));

  CUDA__Safe_Call(cudaMalloc((void **)&tmp_ptr_device_struct_cudaDeviceProp,
                             sizeof(struct cudaDeviceProp)));

  for (; device_id != tmp_number_CUDA_devices; ++device_id) {
    CUDA__Safe_Call(
        cudaGetDeviceProperties(&tmp_struct_cudaDeviceProp, device_id));

    CUDA__Safe_Call(cudaMemcpy(
        tmp_ptr_device_struct_cudaDeviceProp, &tmp_struct_cudaDeviceProp,
        sizeof(struct cudaDeviceProp), cudaMemcpyKind::cudaMemcpyHostToDevice));

    kernel__Dataset_device__Add_CUDA_Device<<<1, 1u>>>(
        device_id, tmp_ptr_device_struct_cudaDeviceProp, this);

    CUDA__Check_Error();
  }

  CUDA__Safe_Call(cudaFree(tmp_ptr_device_struct_cudaDeviceProp));

  return true;
}

template <typename T>
__host__ cuDataset<T>::cuDataset(DL::DATASET_FORMAT::TYPE const
                                     dset_fmt,
                                 std::string const &ref_path_received) {}

template <typename T>
__host__ __device__ cuDataset<T>::~cuDataset(void) {
  this->Deallocate();
}

template <class T>
__device__ cuDataset<T> &cuDataset<T>::operator=(
    class cuDataset<T> const &ref_Dataset_received) {
  if (&ref_Dataset_received != this) {
    this->copy(ref_Dataset_received);
  }

  return *this;
}

template <typename T>
__global__ void kernel__Dataset_device__Initialize(
    DL::DATASET::TYPE const
        type_gradient_descent_received,
    class cuDataset<T> *const ptr_Dataset_device_received) {
  ptr_Dataset_device_received->Initialize(type_gradient_descent_received);
}
template __global__ void kernel__Dataset_device__Initialize(
    DL::DATASET::TYPE const,
    class cuDataset<var> *const);

template <typename T>
__host__ __device__ bool cuDataset<T>::Initialize(
    DL::DATASET::TYPE const type_gradient_descent_received) {
#ifndef COMPILE_CUDA
  kernel__Dataset_device__Initialize<T>
      <<<1, 1u>>>(type_gradient_descent_received, this);

  CUDA__Check_Error();

  return true;
#else
  if (this->p_type_dataset_process ==
      DL::DATASET::NONE) {
    this->p_n_data = 0u;
    this->p_seq_w = 0u;
    this->p_n_inp = 0u;
    this->p_n_out = 0u;

    this->Xm = nullptr;
    this->Ym = nullptr;

    this->p_type_dataset_process = type_gradient_descent_received;

    this->ptr_array_dim3_grid_batch = NULL;
    this->ptr_array_dim3_block_batch = NULL;

    this->ptr_array_dim3_grid_batch_fold = NULL;
    this->ptr_array_dim3_block_batch_fold = NULL;

    this->ptr_array_dim3_grid_shuffle = NULL;
    this->ptr_array_dim3_block_shuffle = NULL;

    this->p_ptr_Class_Device_Information_Array = nullptr;

    // Mini-Batch Stochastic
    this->use_shuffle = true;

    this->p_number_mini_batch = 0u;
    this->p_number_data_mini_batch = 0u;
    this->p_number_data_per_iteration = 0u;
    this->p_number_data_last_iteration = 0u;
    this->ptr_array_stochastic_index = nullptr;

    this->ptr_array_inputs_array_stochastic = nullptr;
    this->ptr_array_outputs_array_stochastic = nullptr;
    // - Mini-Batch Stochastic -

    // Cross Validation k-fold
    this->number_data_k_fold = 0u;
    this->number_k_fold = 0u;
    this->number_k_sub_fold = 0u;
    this->number_data_per_fold = 0u;
    this->number_data_training = 0u;
    this->number_data_validating = 0u;
    this->number_data_per_sub_iteration = 0u;
    this->number_data_last_sub_iteration = 0u;

    this->ptr_array_inputs_array_k_fold = nullptr;
    this->ptr_array_outputs_array_k_fold = nullptr;
    this->ptr_array_inputs_array_k_sub_fold = nullptr;
    this->ptr_array_outputs_array_k_sub_fold = nullptr;

    this->ptr_Validation_Dataset = nullptr;
    // - Cross Validation k-fold -

    // cuRAND.
    this->p_number_cuRAND_State_MTGP32_shuffle = 0u;
    this->p_number_blocks_shuffle = 0u;

    this->ptr_array_cuRAND_State_MTGP32_shuffle = nullptr;
    // |END| cuRAND. |END|
  } else {
    return false;
  }

  return true;
#endif
}

template <typename T>
__global__ void
kernel__Dataset_device__Initialize_Mini_Batch_Stochastic_Gradient_Descent(
    bool const use_shuffle_received,
    size_t const desired_number_data_per_mini_batch_received,
    size_t const number_mini_batch_maximum_received,
    class cuDataset<T> *const ptr_Dataset_device_received) {
  ptr_Dataset_device_received
      ->Initialize_Mini_Batch_Stochastic_Gradient_Descent(
          use_shuffle_received, desired_number_data_per_mini_batch_received,
          number_mini_batch_maximum_received);
}
template __global__ void
kernel__Dataset_device__Initialize_Mini_Batch_Stochastic_Gradient_Descent(
    bool const, size_t const, size_t const, class cuDataset<var> *const);

template <typename T>
__host__ __device__ bool
cuDataset<T>::Initialize_Mini_Batch_Stochastic_Gradient_Descent(
    bool const use_shuffle_received,
    size_t const desired_number_data_per_mini_batch_received,
    size_t const number_mini_batch_maximum_received) {
#ifndef COMPILE_CUDA
  kernel__Dataset_device__Initialize_Mini_Batch_Stochastic_Gradient_Descent<T>
      <<<1, 1u>>>(use_shuffle_received,
                   desired_number_data_per_mini_batch_received,
                   number_mini_batch_maximum_received, this);

  CUDA__Check_Error();

  return true;
#else
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available. At line %d.", __LINE__);

    return false;
  } else if (desired_number_data_per_mini_batch_received == 0u) {
    ERR(
        L"Desired number data per mini-batch equal zero. At line "
        "%d.", __LINE__);

    return false;
  }

  // 34875 / 128 = 272.46
  // 101 / 16 = 6.3125
  double const tmp_number_mini_batch(
      static_cast<double>(this->p_n_data) /
      static_cast<double>(desired_number_data_per_mini_batch_received));

  // 272.46 = 272
  // 6.3125 = 6
  this->p_number_mini_batch = static_cast<size_t>(tmp_number_mini_batch);
  if (number_mini_batch_maximum_received != 0u) {
    this->p_number_mini_batch =
        this->p_number_mini_batch > number_mini_batch_maximum_received
            ? number_mini_batch_maximum_received
            : this->p_number_mini_batch;
  }

  // 128
  // 16
  this->p_number_data_per_iteration =
      desired_number_data_per_mini_batch_received;
  // 128 + (272.46 - 272) * 128 = 187
  // 16 + (6.3125 - 6) * 16 = 21
  this->p_number_data_last_iteration =
      this->p_number_data_per_iteration +
      static_cast<size_t>(
          (tmp_number_mini_batch -
           static_cast<double>(this->p_number_mini_batch)) *
          static_cast<double>(this->p_number_data_per_iteration));

  this->p_number_data_mini_batch = this->p_number_data_last_iteration;

  this->ptr_array_inputs_array_stochastic =
      new T *[this->p_number_data_last_iteration];

  this->ptr_array_outputs_array_stochastic =
      new T *[this->p_number_data_last_iteration];

  this->use_shuffle = use_shuffle_received;

  this->ptr_array_stochastic_index = new size_t[this->p_n_data];
  if (this->ptr_array_stochastic_index == nullptr) {
    ERR(L"Can not allocate %zu bytes. At line %d.", this->p_n_data * sizeof(size_t),
                 __LINE__);

    return false;
  }

  if (use_shuffle_received) {
    this->Mini_Batch_Stochastic__Initialize_Shuffle();
  } else {
    Memory::Memory_Initialize_Index<size_t>(
        this->p_n_data, this->ptr_array_stochastic_index,
        this->ptr_array_dim3_grid_batch, this->ptr_array_dim3_block_batch);
  }

  return true;
#endif
}

template <typename T>
__global__ void kernel__Dataset_device__Initialize_Cross_Validation_K_Fold(
    bool const use_shuffle_received, size_t const number_k_fold_received,
    size_t const number_k_sub_fold_received,
    class cuDatasets<T> const *const ptr_CUDA_Dataset_Manager_received,
    class cuDataset<T> *const ptr_Dataset_device_received) {
  ptr_Dataset_device_received->Initialize__Cross_Validation(
      use_shuffle_received, number_k_fold_received, number_k_sub_fold_received,
      ptr_CUDA_Dataset_Manager_received);
}
template __global__ void
kernel__Dataset_device__Initialize_Cross_Validation_K_Fold(
    bool const, size_t const, size_t const, class cuDatasets<var> const *const,
    class cuDataset<var> *const);

template <typename T>
__host__ __device__ bool cuDataset<T>::Initialize__Cross_Validation(
    bool const use_shuffle_received, size_t const number_k_fold_received,
    size_t const number_k_sub_fold_received,
    class cuDatasets<T> const *const ptr_CUDA_Dataset_Manager_received) {
#ifndef COMPILE_CUDA
  kernel__Dataset_device__Initialize_Cross_Validation_K_Fold<T><<<1, 1u>>>(
      use_shuffle_received, number_k_fold_received, number_k_sub_fold_received,
      ptr_CUDA_Dataset_Manager_received, this);

  CUDA__Check_Error();

  return true;
#else
  if (this->p_n_data == 0_UZ) {
    ERR(L"Amount of data not available.",);

    return false;
  } else if (number_k_fold_received < 2u) {
    ERR(
        L"Not enough K-fold. Need to be at least at 2.",);

    return false;
  } else if (ptr_CUDA_Dataset_Manager_received == nullptr) {
    ERR(
        L"\"ptr_CUDA_Dataset_Manager_received\" is a "
        "nullptr.",);

    return false;
  }

  class cuDataset<T> *const tmp_ptr_Dataset_device_validation(
      ptr_CUDA_Dataset_Manager_received->get_dataset(
          DL::ENV::VALID));

  if (tmp_ptr_Dataset_device_validation == nullptr) {
    ERR(
        L""
        "\"get_dataset(DL::ENV::TYPE_DATASET_"
        "VALIDATION)\" is a nullptr.",);

    return false;
  } else if (this == tmp_ptr_Dataset_device_validation) {
    ERR(
        L"Can not use cross-validation without a testing or "
        "validating set.",);

    return false;
  }

  size_t const tmp_number_data_TnV(
      this->Get__Total_Data() +
      tmp_ptr_Dataset_device_validation->Get__Total_Data()),
      tmp_maximum_number_data(
          ptr_CUDA_Dataset_Manager_received->get_n_data()),
      n_data(
          std::min<T>(tmp_number_data_TnV, tmp_maximum_number_data));

  if (n_data == number_k_fold_received) {
    ERR(
        L"K-fold can not be equal to the amount of data "
        "available.",);

    return false;
  } else if (n_data < number_k_fold_received) {
    ERR(
        L"K-fold can not be larger than the number of data "
        "available.",);

    return false;
  }

  this->copy(*ptr_CUDA_Dataset_Manager_received);

  this->p_n_data = n_data;

  this->Get__Class_Device_Information_Array()
      ->Get__CUDA_Device()
      ->Grid_Block_1Dimensions(n_data, 0,
                               *this->ptr_array_dim3_grid_batch,
                               *this->ptr_array_dim3_block_batch);

  size_t const tmp_number_data_per_fold(this->p_n_data /
                                        number_k_fold_received);

  this->number_k_fold = number_k_fold_received;

  this->number_data_per_fold = tmp_number_data_per_fold;
  this->number_data_training =
      (number_k_fold_received - 1u) * tmp_number_data_per_fold;
  this->number_data_validating =
      this->p_n_data - this->number_data_training;

  if (number_k_sub_fold_received > this->number_data_training) {
    ERR(
        L"K-sub-fold (%u) > (%u) amount of training data.", number_k_sub_fold_received, this->number_data_training);

    return false;
  }

  this->number_k_sub_fold = number_k_sub_fold_received == 0u
                                ? number_k_fold_received - 1u
                                : number_k_sub_fold_received;

  // 8 / 2 = 4
  // 31383 / 240 = 130.7625
  double const tmp_number_data_per_sub_fold(
      static_cast<double>(this->number_data_training) /
      static_cast<double>(this->number_k_sub_fold));

  // 4
  // 130
  this->number_data_per_sub_iteration =
      static_cast<size_t>(tmp_number_data_per_sub_fold);

  // 4 + (4 - 4) * 2 = 0
  // 130 + (130.7625 - 130) * 240 = 183
  this->number_data_last_sub_iteration =
      this->number_data_per_sub_iteration +
      static_cast<size_t>(
          (tmp_number_data_per_sub_fold -
           static_cast<double>(this->number_data_per_sub_iteration)) *
          static_cast<double>(this->number_k_sub_fold));

  // 4 * 1 + 4 = 8
  // 130 * 239 + (130 + 183) = 31383
  this->number_data_k_fold = this->number_data_last_sub_iteration;

  this->Get__Class_Device_Information_Array()
      ->Get__CUDA_Device()
      ->Grid_Block_1Dimensions(this->number_data_training, 0,
                               *this->ptr_array_dim3_grid_batch_fold,
                               *this->ptr_array_dim3_block_batch_fold);

  this->ptr_array_inputs_array_k_fold = new T *[this->number_data_training];

  this->ptr_array_outputs_array_k_fold = new T *[this->number_data_training];

  this->ptr_array_inputs_array_k_sub_fold = this->ptr_array_inputs_array_k_fold;

  this->ptr_array_outputs_array_k_sub_fold =
      this->ptr_array_outputs_array_k_fold;

  this->use_shuffle = use_shuffle_received;

  this->ptr_array_stochastic_index = new size_t[this->p_n_data];
  if (this->ptr_array_stochastic_index == nullptr) {
    ERR(L"Can not allocate %zu bytes. At line %d.", this->p_n_data * sizeof(size_t),
                 __LINE__);

    return false;
  }

  if (use_shuffle_received) {
    this->Cross_Validation_K_Fold__Initialize_Shuffle();
  } else {
    Memory::Memory_Initialize_Index<size_t>(
        this->p_n_data, this->ptr_array_stochastic_index,
        this->ptr_array_dim3_grid_batch, this->ptr_array_dim3_block_batch);
  }

  this->ptr_Validation_Dataset = tmp_ptr_Dataset_device_validation;

  return true;
#endif
}

template <typename T>
__global__ void kernel__Dataset_device__Initialize_Cross_Validation_K_Fold(
    class cuDatasets<T> const *const ptr_CUDA_Dataset_Manager_received,
    class cuDataset<T> *const ptr_Dataset_device_received) {
  ptr_Dataset_device_received->Initialize__Cross_Validation(
      ptr_CUDA_Dataset_Manager_received);
}
template __global__ void
kernel__Dataset_device__Initialize_Cross_Validation_K_Fold(
    class cuDatasets<var> const *const, class cuDataset<var> *const);

template <typename T>
__host__ __device__ bool cuDataset<T>::Initialize__Cross_Validation(
    class cuDatasets<T> const *const ptr_CUDA_Dataset_Manager_received) {
#ifndef COMPILE_CUDA
  kernel__Dataset_device__Initialize_Cross_Validation_K_Fold<T>
      <<<1, 1u>>>(ptr_CUDA_Dataset_Manager_received, this);

  CUDA__Check_Error();

  return true;
#else
  if (this->p_n_data == 0_UZ) {
    ERR(L"Amount of data not available.",);

    return false;
  } else if (ptr_CUDA_Dataset_Manager_received == nullptr) {
    ERR(
        L"\"datasets\" is a nullptr.",);

    return false;
  }

  class cuDataset<T> *const tmp_ptr_Dataset_Cross_Validation_training(
      ptr_CUDA_Dataset_Manager_received->get_dataset(
          DL::ENV::TRAIN));

  if (tmp_ptr_Dataset_Cross_Validation_training == nullptr) {
    ERR(
        L""
        "\"get_dataset(DL::ENV::TYPE_DATASET_"
        "TRAINING)\" is a nullptr.",);

    return false;
  } else if (this == tmp_ptr_Dataset_Cross_Validation_training) {
    ERR(
        L"Can not use cross-validation without a testing or "
        "validating set.",);

    return false;
  }

  this->copy(*ptr_CUDA_Dataset_Manager_received);

  this->p_n_data =
      tmp_ptr_Dataset_Cross_Validation_training->Get__Total_Data();

  this->Get__Class_Device_Information_Array()
      ->Get__CUDA_Device()
      ->Grid_Block_1Dimensions(this->p_n_data, 0,
                               *this->ptr_array_dim3_grid_batch,
                               *this->ptr_array_dim3_block_batch);

  this->number_k_fold =
      tmp_ptr_Dataset_Cross_Validation_training->Get__Number_CV_K_Fold();
  this->number_data_per_fold =
      tmp_ptr_Dataset_Cross_Validation_training->Get__Number_CV_Data_Per_Fold();
  this->number_data_training =
      tmp_ptr_Dataset_Cross_Validation_training->Get__Number_CV_Data_Training();
  this->number_data_validating = tmp_ptr_Dataset_Cross_Validation_training
                                     ->Get__Number_CV_Data_Validating();

  this->number_k_sub_fold =
      tmp_ptr_Dataset_Cross_Validation_training->Get__Number_CV_K_Sub_Fold();
  this->number_data_per_sub_iteration =
      tmp_ptr_Dataset_Cross_Validation_training
          ->Get__Number_CV_Data_Per_Sub_Iteration();
  this->number_data_last_sub_iteration =
      tmp_ptr_Dataset_Cross_Validation_training
          ->Get__Number_CV_Data_Last_Sub_Iteration();

  this->number_data_k_fold = this->number_data_validating;

  this->Get__Class_Device_Information_Array()
      ->Get__CUDA_Device()
      ->Grid_Block_1Dimensions(this->number_data_validating, 0,
                               *this->ptr_array_dim3_grid_batch_fold,
                               *this->ptr_array_dim3_block_batch_fold);

  this->ptr_array_inputs_array_k_fold = new T *[this->number_data_validating];

  this->ptr_array_outputs_array_k_fold = new T *[this->number_data_validating];

  this->ptr_array_inputs_array_k_sub_fold = this->ptr_array_inputs_array_k_fold;

  this->ptr_array_outputs_array_k_sub_fold =
      this->ptr_array_outputs_array_k_fold;

  return true;
#endif
}

template <typename T>
__device__ void cuDataset<T>::Mini_Batch_Stochastic__Reset(void) {
  this->p_number_data_mini_batch = this->p_number_data_last_iteration;
}

template <typename T>
__global__ void kernel__Two_Memory_2D_Copy_Stochastic(
    size_t const *const ptr_array_stochastic_index_received,
    T **const ptr_array_destination_0_received,
    T **const ptr_array_destination_1_received,
    T **const ptr_array_source_0_received,
    T **const ptr_array_source_1_received) {
  size_t const tmp_thread_index(blockIdx.x * blockDim.x + threadIdx.x);

  ptr_array_destination_0_received[tmp_thread_index] =
      ptr_array_source_0_received
          [ptr_array_stochastic_index_received[tmp_thread_index]];
  ptr_array_destination_1_received[tmp_thread_index] =
      ptr_array_source_1_received
          [ptr_array_stochastic_index_received[tmp_thread_index]];
}

template <typename T>
__global__ void kernel__Two_Memory_2D_Copy_Stochastic(
    size_t const size_received,
    size_t const *const ptr_array_stochastic_index_received,
    T **const ptr_array_destination_0_received,
    T **const ptr_array_destination_1_received,
    T **const ptr_array_source_0_received,
    T **const ptr_array_source_1_received) {
  size_t const tmp_thread_index(blockIdx.x * blockDim.x + threadIdx.x);

  if (tmp_thread_index < size_received) {
    ptr_array_destination_0_received[tmp_thread_index] =
        ptr_array_source_0_received
            [ptr_array_stochastic_index_received[tmp_thread_index]];
    ptr_array_destination_1_received[tmp_thread_index] =
        ptr_array_source_1_received
            [ptr_array_stochastic_index_received[tmp_thread_index]];
  }
}

template <typename T>
__global__ void kernel_while__Two_Memory_2D_Copy_Stochastic(
    size_t const size_received,
    size_t const *const ptr_array_stochastic_index_received,
    T **const ptr_array_destination_0_received,
    T **const ptr_array_destination_1_received,
    T **const ptr_array_source_0_received,
    T **const ptr_array_source_1_received) {
  size_t const tmp_grid_stride(gridDim.x * blockDim.x);
  size_t tmp_thread_index(blockIdx.x * blockDim.x + threadIdx.x);

  do {
    ptr_array_destination_0_received[tmp_thread_index] =
        ptr_array_source_0_received
            [ptr_array_stochastic_index_received[tmp_thread_index]];
    ptr_array_destination_1_received[tmp_thread_index] =
        ptr_array_source_1_received
            [ptr_array_stochastic_index_received[tmp_thread_index]];

    tmp_thread_index += tmp_grid_stride;
  } while (tmp_thread_index < size_received);
}

template <typename T>
__device__ void Two_Memory_2D_Copy_Stochastic(
    size_t const size_received,
    size_t const *const ptr_array_stochastic_index_received,
    T **const ptr_array_destination_0_received,
    T **const ptr_array_destination_1_received,
    T **const ptr_array_source_0_received,
    T **const ptr_array_source_1_received,
    struct dim3 const *const ptr_dimension_grid_received,
    struct dim3 const *const ptr_dimension_block_received) {
  if (USE_PARALLEL && size_received >= warpSize) {
    LAUNCH_KERNEL_POINTER_1D(
        Two_Memory_2D_Copy_Stochastic<T>, ptr_dimension_grid_received,
        ptr_dimension_block_received, 0_UZ, size_received,
        ptr_array_stochastic_index_received, ptr_array_destination_0_received,
        ptr_array_destination_1_received, ptr_array_source_0_received,
        ptr_array_source_1_received)
  } else {
    for (size_t i(0_UZ); i != size_received; ++i) {
      ptr_array_destination_0_received[i] =
          ptr_array_source_0_received[ptr_array_stochastic_index_received[i]];

      ptr_array_destination_1_received[i] =
          ptr_array_source_1_received[ptr_array_stochastic_index_received[i]];
    }
  }
}

template <typename T>
__device__ bool cuDataset<T>::Mini_Batch_Stochastic__Increment_Mini_Batch(
    size_t const mini_batch_iteration_received) {
  size_t const tmp_data_per_mini_batch(
      mini_batch_iteration_received + 1u != this->p_number_mini_batch
          ? this->p_number_data_per_iteration
          : this->p_number_data_last_iteration);
  size_t tmp_last_element_start_index, tmp_last_element_end_index;

  tmp_last_element_start_index =
      mini_batch_iteration_received * this->p_number_data_per_iteration;
  tmp_last_element_end_index =
      tmp_last_element_start_index + tmp_data_per_mini_batch;

  // Index global inputs to local inputs.
  Two_Memory_2D_Copy_Stochastic<T>(
      tmp_last_element_end_index - tmp_last_element_start_index,
      this->ptr_array_stochastic_index + tmp_last_element_start_index,
      this->ptr_array_inputs_array_stochastic,
      this->ptr_array_outputs_array_stochastic, this->Xm,
      this->Ym, this->ptr_array_dim3_grid_batch,
      this->ptr_array_dim3_block_batch);
  // |END| Index global inputs to local inputs. |END|

  this->p_number_data_mini_batch = tmp_data_per_mini_batch;

  // Do we need to synchronise? Based on "Two_Memory_2D_Copy_Stochastic"
  // Function.
  // => Synchronisation before using the training mini-batch.
  if (tmp_last_element_end_index - tmp_last_element_start_index >= warpSize) {
    CUDA__Check_Error();
  }

  return true;
}

template <typename T>
__device__ var cuDataset<T>::Test_Epoch_Cross_Validation_K_Fold(
    class cuModel *ptr_cuModel_received) {
  ptr_cuModel_received->reset_loss();

  ptr_cuModel_received->type_state_propagation = DL::
      PROPAGATION::INFERENCE;

  size_t const n_data(this->number_data_k_fold),
      tmp_maximum_batch_size(ptr_cuModel_received->batch_size),
      tmp_number_batchs(static_cast<size_t>(
          ceil(static_cast<double>(n_data) /
               static_cast<double>(tmp_maximum_batch_size))));
  size_t tmp_batch_size, i;

  for (i = 0u; i != tmp_number_batchs; ++i) {
    tmp_batch_size = i + 1u != tmp_number_batchs
                         ? tmp_maximum_batch_size
                         : n_data - i * tmp_maximum_batch_size;

    ptr_cuModel_received->forward_pass(
        tmp_batch_size,
        this->ptr_array_inputs_array_k_fold + i * tmp_maximum_batch_size);

    ptr_cuModel_received->Test(
        tmp_batch_size,
        this->ptr_array_outputs_array_k_fold + i * tmp_maximum_batch_size);
  }

  *ptr_cuModel_received->ptr_array_number_loss =
      n_data * this->get_n_out();
  ptr_cuModel_received->n_acc_trial =
      n_data * this->get_n_out();

  // Synchronize the computed error before merging between threads.
  CUDA__Check_Error();

  ptr_cuModel_received->merge_mp_accu_loss();

  ptr_cuModel_received->type_state_propagation = DL::
      PROPAGATION::TRAINING;

  return (ptr_cuModel_received->get_loss(
      DL::ENV::NONE));
}

template <typename T>
__device__ void cuDataset<T>::Cross_Validation_K_Fold__Initialize_Shuffle(
    void) {
  class cuDeviceProp const *const tmp_ptr_CUDA_Device(
      this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

  // Tree shift shuffle.
  if (this->ptr_array_dim3_grid_shuffle == NULL) {
    struct dim3 *tmp_ptr_array_dim3_grid_shuffle(
        static_cast<struct dim3 *>(malloc(sizeof(struct dim3))));
    if (tmp_ptr_array_dim3_grid_shuffle == NULL) {
      ERR(
          L"Can not allocate memory. malloc(sizeof(%u))", sizeof(struct dim3));

      return;
    }
    *tmp_ptr_array_dim3_grid_shuffle = dim3(1, 1, 1u);
    this->ptr_array_dim3_grid_shuffle = tmp_ptr_array_dim3_grid_shuffle;
  }

  if (this->ptr_array_dim3_block_shuffle == NULL) {
    struct dim3 *tmp_ptr_array_dim3_block_shuffle(
        static_cast<struct dim3 *>(malloc(sizeof(struct dim3))));
    if (tmp_ptr_array_dim3_block_shuffle == NULL) {
      ERR(
          L"Can not allocate memory. malloc(sizeof(%u))", sizeof(struct dim3));

      return;
    }
    *tmp_ptr_array_dim3_block_shuffle = dim3(1, 1, 1u);
    this->ptr_array_dim3_block_shuffle = tmp_ptr_array_dim3_block_shuffle;
  }

  this->p_number_blocks_shuffle = static_cast<size_t>(
      ceil(static_cast<double>(this->p_n_data) /
           static_cast<double>(this->Get__Class_Device_Information_Array()
                                   ->Get__CUDA_Device()
                                   ->Get__Warp_Size())));

  tmp_ptr_CUDA_Device->Grid_Block_cuRAND_1Dimensions(
      this->p_number_blocks_shuffle, 0, this->ptr_array_dim3_grid_shuffle[0],
      this->ptr_array_dim3_block_shuffle[0]);
  // |END| Tree shift shuffle. |END|
}

template <typename T>
__device__ void cuDataset<T>::Cross_Validation_K_Fold__Shuffle(void) {
  Memory::Memory_Initialize_Index_Shift<size_t>(
      this->p_n_data,
      curand(this->ptr_array_cuRAND_State_MTGP32_shuffle) %
          this->p_n_data,
      this->ptr_array_stochastic_index, this->ptr_array_dim3_grid_batch,
      this->ptr_array_dim3_block_batch);

  Shuffle::Tree_Shuffle<size_t>(
      this->p_number_blocks_shuffle,
      this->Get__Class_Device_Information_Array()
          ->Get__CUDA_Device()
          ->Get__Warp_Size(),
      this->p_n_data, this->ptr_array_stochastic_index,
      this->ptr_array_cuRAND_State_MTGP32_shuffle,
      this->ptr_array_dim3_grid_shuffle, this->ptr_array_dim3_block_shuffle);
}

template <typename T>
__device__ void cuDataset<T>::Cross_Validation_K_Fold__Reset(void) {
  this->ptr_array_inputs_array_k_sub_fold = this->ptr_array_inputs_array_k_fold;

  this->ptr_array_outputs_array_k_sub_fold =
      this->ptr_array_outputs_array_k_fold;

  this->number_data_k_fold = this->number_data_last_sub_iteration;
}

template <typename T>
__device__ bool cuDataset<T>::Cross_Validation_K_Fold__Increment_Fold(
    size_t const fold_received) {
  if (this->p_n_data == 0_UZ) {
    ERR(L"Amount of data not available.",);

    return false;
  }

  bool tmp_synchronized(true);

  if (fold_received >= this->number_k_fold) {
    return false;
  }

  size_t const tmp_number_data_training_per_fold(this->number_data_per_fold),
      tmp_number_data_validating(
          this->ptr_Validation_Dataset->number_data_validating),
      tmp_validating_index_start(fold_received *
                                 tmp_number_data_training_per_fold),
      tmp_validating_index_end(tmp_validating_index_start +
                               tmp_number_data_validating);
  size_t *tmp_ptr_array_stochastic_index(this->ptr_array_stochastic_index);

  if (tmp_validating_index_start == 0u)  // First iteration.
  {
    // Validation sample.
    // (0, 1, 2)   [3, 4, 5   6, 7, 8   9, 10, 11]
    Two_Memory_2D_Copy_Stochastic<T>(
        tmp_number_data_validating, tmp_ptr_array_stochastic_index,
        this->ptr_Validation_Dataset->ptr_array_inputs_array_k_fold,
        this->ptr_Validation_Dataset->ptr_array_outputs_array_k_fold,
        this->Xm, this->Ym,
        this->ptr_Validation_Dataset->ptr_array_dim3_grid_batch_fold,
        this->ptr_Validation_Dataset->ptr_array_dim3_block_batch_fold);

    // Do we need to synchronise? Based on "Two_Memory_2D_Copy_Stochastic"
    // Function.
    if (tmp_number_data_validating >= warpSize) {
      tmp_synchronized = false;
    }
    // |END| Validation sample. |END|

    // Training sample.
    tmp_ptr_array_stochastic_index += tmp_validating_index_end;

    // (0, 1, 2)   [3, 4, 5   6, 7, 8   9, 10, 11]
    Two_Memory_2D_Copy_Stochastic<T>(
        this->number_data_training, tmp_ptr_array_stochastic_index,
        this->ptr_array_inputs_array_k_fold,
        this->ptr_array_outputs_array_k_fold, this->Xm,
        this->Ym, this->ptr_array_dim3_grid_batch_fold,
        this->ptr_array_dim3_block_batch_fold);

    // Do we need to synchronise? Based on "Two_Memory_2D_Copy_Stochastic"
    // Function.
    if (this->number_data_training >= warpSize) {
      tmp_synchronized = false;
    }
    // |END| Training sample. |END|
  } else if (tmp_validating_index_end ==
             this->p_n_data)  // Last iteration.
  {
    // Training sample.
    // [0, 1, 2   3, 4, 5   6, 7, 8]   (9, 10, 11)
    Two_Memory_2D_Copy_Stochastic<T>(
        this->number_data_training, tmp_ptr_array_stochastic_index,
        this->ptr_array_inputs_array_k_fold,
        this->ptr_array_outputs_array_k_fold, this->Xm,
        this->Ym, this->ptr_array_dim3_grid_batch_fold,
        this->ptr_array_dim3_block_batch_fold);

    // Do we need to synchronise? Based on "Two_Memory_2D_Copy_Stochastic"
    // Function.
    if (this->number_data_training >= warpSize) {
      tmp_synchronized = false;
    }
    // |END| Training sample. |END|

    // Validation sample.
    tmp_ptr_array_stochastic_index += tmp_validating_index_start;

    // [0, 1, 2   3, 4, 5   6, 7, 8]   (9, 10, 11)
    Two_Memory_2D_Copy_Stochastic<T>(
        tmp_number_data_validating, tmp_ptr_array_stochastic_index,
        this->ptr_Validation_Dataset->ptr_array_inputs_array_k_fold,
        this->ptr_Validation_Dataset->ptr_array_outputs_array_k_fold,
        this->Xm, this->Ym,
        this->ptr_Validation_Dataset->ptr_array_dim3_grid_batch_fold,
        this->ptr_Validation_Dataset->ptr_array_dim3_block_batch_fold);

    // Do we need to synchronise? Based on "Two_Memory_2D_Copy_Stochastic"
    // Function.
    if (tmp_number_data_validating >= warpSize) {
      tmp_synchronized = false;
    }
    // |END| Validation sample. |END|
  } else  // The remaining iterations.
  {
    // Training sample.
    // [0, 1, 2]   (3, 4, 5)   [6, 7, 8   9, 10, 11]
    Two_Memory_2D_Copy_Stochastic<T>(
        tmp_validating_index_start, tmp_ptr_array_stochastic_index,
        this->ptr_array_inputs_array_k_fold,
        this->ptr_array_outputs_array_k_fold, this->Xm,
        this->Ym, this->ptr_array_dim3_grid_batch_fold,
        this->ptr_array_dim3_block_batch_fold);

    // Do we need to synchronise? Based on "Two_Memory_2D_Copy_Stochastic"
    // Function.
    if (tmp_validating_index_start >= warpSize) {
      tmp_synchronized = false;
    }
    // |END| Training sample. |END|

    // Validation sample.
    tmp_ptr_array_stochastic_index += tmp_validating_index_start;

    // [0, 1, 2]   (3, 4, 5)   [6, 7, 8   9, 10, 11]
    Two_Memory_2D_Copy_Stochastic<T>(
        tmp_number_data_validating, tmp_ptr_array_stochastic_index,
        this->ptr_Validation_Dataset->ptr_array_inputs_array_k_fold,
        this->ptr_Validation_Dataset->ptr_array_outputs_array_k_fold,
        this->Xm, this->Ym,
        this->ptr_Validation_Dataset->ptr_array_dim3_grid_batch_fold,
        this->ptr_Validation_Dataset->ptr_array_dim3_block_batch_fold);

    // Do we need to synchronise? Based on "Two_Memory_2D_Copy_Stochastic"
    // Function.
    if (tmp_number_data_validating >= warpSize) {
      tmp_synchronized = false;
    }
    // |END| Validation sample. |END|

    // Training sample.
    tmp_ptr_array_stochastic_index =
        this->ptr_array_stochastic_index + tmp_number_data_validating;

    // [0, 1, 2]   (3, 4, 5)   [6, 7, 8   9, 10, 11]
    Two_Memory_2D_Copy_Stochastic<T>(
        this->number_data_training - tmp_validating_index_start,
        tmp_ptr_array_stochastic_index + tmp_validating_index_start,
        this->ptr_array_inputs_array_k_fold + tmp_validating_index_start,
        this->ptr_array_outputs_array_k_fold + tmp_validating_index_start,
        this->Xm, this->Ym,
        this->ptr_array_dim3_grid_batch_fold,
        this->ptr_array_dim3_block_batch_fold);

    // Do we need to synchronise? Based on "Two_Memory_2D_Copy_Stochastic"
    // Function.
    if (this->number_data_training - tmp_validating_index_start >= warpSize) {
      tmp_synchronized = false;
    }
    // |END| Training sample. |END|
  }

  // Do we need to synchronise? Based on "Two_Memory_2D_Copy_Stochastic"
  // Function.
  // => Synchronisation before using the training fold batch.
  if (tmp_synchronized == false) {
    CUDA__Check_Error();
  }

  return true;
}

template <typename T>
__device__ bool cuDataset<T>::Cross_Validation_K_Fold__Increment_Sub_Fold(
    size_t const sub_fold_received) {
  if (this->number_k_sub_fold == 1u) {
    return true;
  } else if (sub_fold_received >= this->number_k_sub_fold) {
    return false;
  }

  size_t const tmp_data_per_sub_fold(
      sub_fold_received + 1u != this->number_k_sub_fold
          ? this->number_data_per_sub_iteration
          : this->number_data_last_sub_iteration);

  this->ptr_array_inputs_array_k_sub_fold =
      this->ptr_array_inputs_array_k_fold +
      sub_fold_received * this->number_data_per_sub_iteration;

  this->ptr_array_outputs_array_k_sub_fold =
      this->ptr_array_outputs_array_k_fold +
      sub_fold_received * this->number_data_per_sub_iteration;

  this->number_data_k_fold = tmp_data_per_sub_fold;

  return true;
}

template <typename T>
__global__ void kernel__Dataset_device__Set__Type_Gradient_Descent(
    DL::DATASET::TYPE const
        type_gradient_descent_received,
    class cuDataset<T> *const ptr_Dataset_device_received) {
  ptr_Dataset_device_received->Set__Type_Gradient_Descent(
      type_gradient_descent_received);
}
template __global__ void kernel__Dataset_device__Set__Type_Gradient_Descent(
    DL::DATASET::TYPE const,
    class cuDataset<var> *const);

template <typename T>
__host__ __device__ bool cuDataset<T>::Set__Type_Gradient_Descent(
    DL::DATASET::TYPE const type_gradient_descent_received) {
#ifndef COMPILE_CUDA
  kernel__Dataset_device__Set__Type_Gradient_Descent<T>
      <<<1, 1u>>>(type_gradient_descent_received, this);

  CUDA__Check_Error();

  return true;
#else
  this->p_type_dataset_process = type_gradient_descent_received;

  return true;
#endif
}

template <typename T>
__device__ void cuDataset<T>::copy(
    class cuDataset<T> const &ref_Dataset_received) {
  if (this->_reference == false) {
    this->Deallocate();
  }

  this->p_n_data = ref_Dataset_received.p_n_data;
  this->p_n_inp = ref_Dataset_received.p_n_inp;
  this->p_n_out = ref_Dataset_received.p_n_out;
  this->p_seq_w =
      ref_Dataset_received.p_seq_w;

  this->Xm =
      ref_Dataset_received.Xm;
  this->Ym =
      ref_Dataset_received.Ym;

  this->_reference = true;
}

template <typename T>
__device__ void cuDataset<T>::copy(
    class cuDatasets<T> const &ref_Dataset_Manager_received) {
  if (this->_reference == false) {
    this->Deallocate();
  }

  this->p_n_data = ref_Dataset_Manager_received.get_n_data();
  this->p_n_inp = ref_Dataset_Manager_received.get_n_inp();
  this->p_n_out = ref_Dataset_Manager_received.get_n_out();
  this->p_seq_w =
      ref_Dataset_Manager_received.get_seq_w();

  this->Xm =
      const_cast<T **>(ref_Dataset_Manager_received.Get__Input_Array());
  this->Ym =
      const_cast<T **>(ref_Dataset_Manager_received.Get__Output_Array());

  this->_reference = true;
}

template <typename T>
__device__ bool cuDataset<T>::Allocate_Dim3(void) {
  // allocate dim3 batch.
  if (this->ptr_array_dim3_grid_batch == NULL) {
    struct dim3 *tmp_ptr_array_dim3_grid_batch(
        static_cast<struct dim3 *>(malloc(sizeof(struct dim3))));
    if (tmp_ptr_array_dim3_grid_batch == NULL) {
      ERR(
          L"Can not allocate memory. malloc(sizeof(%u))", sizeof(struct dim3));

      return false;
    }
    *tmp_ptr_array_dim3_grid_batch = struct dim3(1, 1, 1u);
    this->ptr_array_dim3_grid_batch = tmp_ptr_array_dim3_grid_batch;
  }

  if (this->ptr_array_dim3_block_batch == NULL) {
    struct dim3 *tmp_ptr_array_dim3_block_batch(
        static_cast<struct dim3 *>(malloc(sizeof(struct dim3))));
    if (tmp_ptr_array_dim3_block_batch == NULL) {
      ERR(
          L"Can not allocate memory. malloc(sizeof(%u))", sizeof(struct dim3));

      return false;
    }
    *tmp_ptr_array_dim3_block_batch = struct dim3(1, 1, 1u);
    this->ptr_array_dim3_block_batch = tmp_ptr_array_dim3_block_batch;
  }
  // |END| allocate dim3 batch. |END|

  // allocate dim3 batch.
  if (this->ptr_array_dim3_grid_batch_fold == NULL) {
    struct dim3 *tmp_ptr_array_dim3_grid_batch_fold(
        static_cast<struct dim3 *>(malloc(sizeof(struct dim3))));
    if (tmp_ptr_array_dim3_grid_batch_fold == NULL) {
      ERR(
          L"Can not allocate memory. malloc(sizeof(%u))", sizeof(struct dim3));

      return false;
    }
    *tmp_ptr_array_dim3_grid_batch_fold = struct dim3(1, 1, 1u);
    this->ptr_array_dim3_grid_batch_fold = tmp_ptr_array_dim3_grid_batch_fold;
  }

  if (this->ptr_array_dim3_block_batch_fold == NULL) {
    struct dim3 *tmp_ptr_array_dim3_block_batch_fold(
        static_cast<struct dim3 *>(malloc(sizeof(struct dim3))));
    if (tmp_ptr_array_dim3_block_batch_fold == NULL) {
      ERR(
          L"Can not allocate memory. malloc(sizeof(%u))", sizeof(struct dim3));

      return false;
    }
    *tmp_ptr_array_dim3_block_batch_fold = struct dim3(1, 1, 1u);
    this->ptr_array_dim3_block_batch_fold = tmp_ptr_array_dim3_block_batch_fold;
  }
  // |END| allocate dim3 batch. |END|

  return true;
}

template <typename T>
__device__ void cuDataset<T>::reference(
    size_t const number_data_received, size_t const number_inputs_received,
    size_t const number_outputs_received,
    size_t const number_recurrent_depth_received,
    T **const ptr_array_inputs_array_received,
    T **const ptr_array_outputs_array_received,
    size_t const number_cuRAND_State_MTGP32_shuffle_received,
    struct curandStateMtgp32 *const ptr_cuRAND_State_MTGP32_received,
    class cuDevicesProp *const ptr_Class_Device_Information_Array_received) {
  this->Deallocate();

  if (this->Allocate_Dim3() == false) {
    ERR(
        L"An error has been triggered from the \"Allocate_Dim3()\" "
        "function. At line %d.", __LINE__);

    return;
  }

  this->p_n_data = number_data_received;
  this->p_n_inp = number_inputs_received;
  this->p_n_out = number_outputs_received;
  this->p_seq_w = number_recurrent_depth_received;

  this->Xm = ptr_array_inputs_array_received;
  this->Ym = ptr_array_outputs_array_received;

  this->p_number_cuRAND_State_MTGP32_shuffle =
      number_cuRAND_State_MTGP32_shuffle_received;

  this->ptr_array_cuRAND_State_MTGP32_shuffle =
      ptr_cuRAND_State_MTGP32_received;

  this->p_ptr_Class_Device_Information_Array =
      ptr_Class_Device_Information_Array_received;

  this->Get__Class_Device_Information_Array()
      ->Get__CUDA_Device()
      ->Grid_Block_1Dimensions(number_data_received, 0,
                               *this->ptr_array_dim3_grid_batch,
                               *this->ptr_array_dim3_block_batch);

  this->_reference = true;
}

template <typename T>
__device__ void cuDataset<T>::Train_Epoch_Batch(
    class cuModel *const ptr_cuModel_received) {
  if (ptr_cuModel_received->use_Dropout) {
    ptr_cuModel_received->Dropout();
  }

  ptr_cuModel_received->reset_loss();

  switch (ptr_cuModel_received->type_optimizer_function) {
    case DL::OPTIMIZER::GD:
    case DL::OPTIMIZER::QUICKPROP:
    case DL::OPTIMIZER::SARPROP:
    case DL::OPTIMIZER::ADAM:
    case DL::OPTIMIZER::ADAMAX:
    case DL::OPTIMIZER::AMSGRAD:
    case DL::OPTIMIZER::NOSADAM:
      this->Train_Batch_Batch(ptr_cuModel_received);
      break;
    case DL::OPTIMIZER::
        IRPROP_MINUS:
    case DL::OPTIMIZER::
        IRPROP_PLUS:
      ptr_cuModel_received->loss_rprop_tm1 =
          ptr_cuModel_received->loss_rprop;

      this->Train_Batch_Batch(ptr_cuModel_received);

      ptr_cuModel_received->loss_rprop =
          abs(ptr_cuModel_received->get_loss(
              DL::ENV::NONE));
      break;
    default:
      ERR(
          L"Undefined optimizer function type (%d).", ptr_cuModel_received->type_optimizer_function);
      break;
  }

  ptr_cuModel_received->merge_mp_accu_loss();
}

template <typename T>
__device__ void cuDataset<T>::Train_Epoch_Mini_Batch_Stochastic(
    class cuModel *const ptr_cuModel_received) {
  if (ptr_cuModel_received->use_Dropout) {
    ptr_cuModel_received->Dropout();
  }

  ptr_cuModel_received->reset_loss();

  switch (ptr_cuModel_received->type_optimizer_function) {
    case DL::OPTIMIZER::GD:
    case DL::OPTIMIZER::QUICKPROP:
    case DL::OPTIMIZER::SARPROP:
    case DL::OPTIMIZER::ADAM:
    case DL::OPTIMIZER::ADAMAX:
    case DL::OPTIMIZER::AMSGRAD:
    case DL::OPTIMIZER::NOSADAM:
      this->Train_Batch_Mini_Batch_Stochastic(ptr_cuModel_received);
      break;
    case DL::OPTIMIZER::
        IRPROP_MINUS:
    case DL::OPTIMIZER::
        IRPROP_PLUS:
      ptr_cuModel_received->loss_rprop_tm1 =
          ptr_cuModel_received->loss_rprop;

      this->Train_Batch_Mini_Batch_Stochastic(ptr_cuModel_received);

      ptr_cuModel_received->loss_rprop =
          abs(ptr_cuModel_received->get_loss(
              DL::ENV::NONE));
      break;
    default:
      ERR(
          L"Undefined optimizer function type (%d).", ptr_cuModel_received->type_optimizer_function);
      break;
  }

  ptr_cuModel_received->merge_mp_accu_loss();
}

template <typename T>
__device__ void cuDataset<T>::Train_Epoch_Cross_Validation_K_Fold(
    class cuModel *const ptr_cuModel_received) {
  if (ptr_cuModel_received->use_Dropout) {
    ptr_cuModel_received->Dropout();
  }

  ptr_cuModel_received->reset_loss();

  switch (ptr_cuModel_received->type_optimizer_function) {
    case DL::OPTIMIZER::GD:
    case DL::OPTIMIZER::QUICKPROP:
    case DL::OPTIMIZER::SARPROP:
    case DL::OPTIMIZER::ADAM:
    case DL::OPTIMIZER::ADAMAX:
    case DL::OPTIMIZER::AMSGRAD:
    case DL::OPTIMIZER::NOSADAM:
      this->Train_Batch_Cross_Validation_K_Fold(ptr_cuModel_received);
      break;
    case DL::OPTIMIZER::
        IRPROP_MINUS:
    case DL::OPTIMIZER::
        IRPROP_PLUS:
      ptr_cuModel_received->loss_rprop_tm1 =
          ptr_cuModel_received->loss_rprop;

      this->Train_Batch_Cross_Validation_K_Fold(ptr_cuModel_received);

      ptr_cuModel_received->loss_rprop =
          abs(ptr_cuModel_received->get_loss(
              DL::ENV::NONE));
      break;
    default:
      ERR(
          L"Undefined optimizer function type (%d).", ptr_cuModel_received->type_optimizer_function);
      break;
  }

  ptr_cuModel_received->merge_mp_accu_loss();
}

template <typename T>
__device__ void cuDataset<T>::Train_Batch_Batch(
    class cuModel *const ptr_cuModel_received) {
  size_t const n_data(this->get_n_data()),
      tmp_maximum_batch_size(ptr_cuModel_received->batch_size),
      tmp_number_batchs(static_cast<size_t>(
          ceil(static_cast<double>(n_data) /
               static_cast<double>(tmp_maximum_batch_size))));
  size_t tmp_batch_size, tmp_batch_index;

  for (tmp_batch_index = 0u; tmp_batch_index != tmp_number_batchs;
       ++tmp_batch_index) {
    tmp_batch_size =
        tmp_batch_index + 1u != tmp_number_batchs
            ? tmp_maximum_batch_size
            : n_data - tmp_batch_index * tmp_maximum_batch_size;

    ptr_cuModel_received->forward_pass(
        tmp_batch_size,
        this->Get__Input_Array() + tmp_batch_index * tmp_maximum_batch_size);

    ptr_cuModel_received->compute_error(
        tmp_batch_size,
        this->Get__Output_Array() + tmp_batch_index * tmp_maximum_batch_size);

    ptr_cuModel_received->backward_pass(tmp_batch_size);

    ptr_cuModel_received->update_derivatives(tmp_batch_size);
  }

  *ptr_cuModel_received->ptr_array_number_loss =
      n_data * this->get_n_out();
  ptr_cuModel_received->n_acc_trial =
      n_data * this->get_n_out();
}

template <typename T>
__device__ void cuDataset<T>::Train_Batch_Mini_Batch_Stochastic(
    class cuModel *const ptr_cuModel_received) {
  size_t const n_data(this->p_number_data_mini_batch),
      tmp_maximum_batch_size(ptr_cuModel_received->batch_size),
      tmp_number_batchs(static_cast<size_t>(
          ceil(static_cast<double>(n_data) /
               static_cast<double>(tmp_maximum_batch_size))));
  size_t tmp_batch_size, i;

  for (i = 0u; i != tmp_number_batchs; ++i) {
    tmp_batch_size = i + 1u != tmp_number_batchs
                         ? tmp_maximum_batch_size
                         : n_data - i * tmp_maximum_batch_size;

    ptr_cuModel_received->forward_pass(
        tmp_batch_size,
        this->ptr_array_inputs_array_stochastic + i * tmp_maximum_batch_size);

    ptr_cuModel_received->compute_error(
        tmp_batch_size,
        this->ptr_array_outputs_array_stochastic + i * tmp_maximum_batch_size);

    ptr_cuModel_received->backward_pass(tmp_batch_size);

    ptr_cuModel_received->update_derivatives(tmp_batch_size);
  }

  *ptr_cuModel_received->ptr_array_number_loss =
      n_data * this->get_n_out();
  ptr_cuModel_received->n_acc_trial =
      n_data * this->get_n_out();
}

template <typename T>
__device__ void cuDataset<T>::Train_Batch_Cross_Validation_K_Fold(
    class cuModel *const ptr_cuModel_received) {
  size_t const n_data(this->number_data_k_fold),
      tmp_maximum_batch_size(ptr_cuModel_received->batch_size),
      tmp_number_batchs(static_cast<size_t>(
          ceil(static_cast<double>(n_data) /
               static_cast<double>(tmp_maximum_batch_size))));
  size_t tmp_batch_size, i;

  for (i = 0u; i != tmp_number_batchs; ++i) {
    tmp_batch_size = i + 1u != tmp_number_batchs
                         ? tmp_maximum_batch_size
                         : n_data - i * tmp_maximum_batch_size;

    ptr_cuModel_received->forward_pass(
        tmp_batch_size,
        this->ptr_array_inputs_array_k_sub_fold + i * tmp_maximum_batch_size);

    ptr_cuModel_received->compute_error(
        tmp_batch_size,
        this->ptr_array_outputs_array_k_sub_fold + i * tmp_maximum_batch_size);

    ptr_cuModel_received->backward_pass(tmp_batch_size);

    ptr_cuModel_received->update_derivatives(tmp_batch_size);
  }

  *ptr_cuModel_received->ptr_array_number_loss =
      n_data * this->get_n_out();
  ptr_cuModel_received->n_acc_trial =
      n_data * this->get_n_out();
}

template <typename T>
__global__ void kernel__Dataset_device__Allocate(
    size_t const number_data_received, size_t const number_inputs_received,
    size_t const number_outputs_received,
    size_t const number_recurrent_depth_received,
    T *const ptr_array_inputs_received, T *const ptr_array_outputs_received,
    class cuDeviceProp *const ptr_Class_Device_Information_received,
    class cuDataset<T> *const ptr_Dataset_device_received) {
  ptr_Dataset_device_received->device_Allocate(
      number_data_received, number_inputs_received, number_outputs_received,
      number_recurrent_depth_received, ptr_array_inputs_received,
      ptr_array_outputs_received, ptr_Class_Device_Information_received);
}

template <typename T>
__global__ void kernel__Two_Memory_Assign_1D_to_2D(
    size_t const step_source_0_received, size_t const step_source_1_received,
    T **const ptr_array_destination_0_received,
    T **const ptr_array_destination_1_received,
    T *const ptr_array_source_0_received,
    T *const ptr_array_source_1_received) {
  size_t const tmp_thread_index(blockIdx.x * blockDim.x + threadIdx.x);

  ptr_array_destination_0_received[tmp_thread_index] =
      ptr_array_source_0_received + tmp_thread_index * step_source_0_received;

  ptr_array_destination_1_received[tmp_thread_index] =
      ptr_array_source_1_received + tmp_thread_index * step_source_1_received;
}

template <typename T>
__global__ void kernel__Two_Memory_Assign_1D_to_2D(
    size_t const size_received, size_t const step_source_0_received,
    size_t const step_source_1_received,
    T **const ptr_array_destination_0_received,
    T **const ptr_array_destination_1_received,
    T *const ptr_array_source_0_received,
    T *const ptr_array_source_1_received) {
  size_t const tmp_thread_index(blockIdx.x * blockDim.x + threadIdx.x);

  if (tmp_thread_index < size_received) {
    ptr_array_destination_0_received[tmp_thread_index] =
        ptr_array_source_0_received + tmp_thread_index * step_source_0_received;

    ptr_array_destination_1_received[tmp_thread_index] =
        ptr_array_source_1_received + tmp_thread_index * step_source_1_received;
  }
}

template <typename T>
__global__ void kernel_while__Two_Memory_Assign_1D_to_2D(
    size_t const size_received, size_t const step_source_0_received,
    size_t const step_source_1_received,
    T **const ptr_array_destination_0_received,
    T **const ptr_array_destination_1_received,
    T *const ptr_array_source_0_received,
    T *const ptr_array_source_1_received) {
  size_t const tmp_grid_stride(gridDim.x * blockDim.x);
  size_t tmp_thread_index(blockIdx.x * blockDim.x + threadIdx.x);

  do {
    ptr_array_destination_0_received[tmp_thread_index] =
        ptr_array_source_0_received + tmp_thread_index * step_source_0_received;

    ptr_array_destination_1_received[tmp_thread_index] =
        ptr_array_source_1_received + tmp_thread_index * step_source_1_received;

    tmp_thread_index += tmp_grid_stride;
  } while (tmp_thread_index < size_received);
}

template <typename T>
__device__ void Two_Memory_Assign_1D_to_2D(
    size_t const size_received, size_t const step_source_0_received,
    size_t const step_source_1_received,
    T **const ptr_array_destination_0_received,
    T **const ptr_array_destination_1_received,
    T *const ptr_array_source_0_received, T *const ptr_array_source_1_received,
    struct dim3 const *const ptr_dimension_grid_received,
    struct dim3 const *const ptr_dimension_block_received) {
  if (USE_PARALLEL && size_received >= warpSize) {
    LAUNCH_KERNEL_POINTER_1D(
        Two_Memory_Assign_1D_to_2D<T>, ptr_dimension_grid_received,
        ptr_dimension_block_received, 0_UZ, size_received,
        step_source_0_received, step_source_1_received,
        ptr_array_destination_0_received, ptr_array_destination_1_received,
        ptr_array_source_0_received, ptr_array_source_1_received)
  } else {
    for (size_t i(0_UZ); i != size_received; ++i) {
      ptr_array_destination_0_received[i] =
          ptr_array_source_0_received + i * step_source_0_received;

      ptr_array_destination_1_received[i] =
          ptr_array_source_1_received + i * step_source_1_received;
    }
  }
}

template <typename T>
__device__ bool cuDataset<T>::device_Allocate(
    size_t const number_data_received, size_t const number_inputs_received,
    size_t const number_outputs_received,
    size_t const number_recurrent_depth_received,
    T const *ptr_array_inputs_received, T const *ptr_array_outputs_received,
    class cuDeviceProp *const ptr_Class_Device_Information_received) {
  T *tmp_ptr_array_inputs, *tmp_ptr_array_outputs;

  this->p_n_data = number_data_received;

  if (this->Allocate_Dim3() == false) {
    ERR(
        L"An error has been triggered from the \"Allocate_Dim3()\" "
        "function. At line %d.", __LINE__);

    return false;
  }

  this->p_n_inp = number_inputs_received;
  this->p_n_out = number_outputs_received;
  this->p_seq_w = number_recurrent_depth_received;

  this->Xm = new T *[number_data_received];
  if (this->Xm == nullptr) {
    ERR(L"Can not allocate %zu bytes. At line %d.",
                 static_cast<size_t>(number_data_received) * sizeof(T *),
                 __LINE__);

    this->Deallocate();

    return false;
  }

  this->Ym = new T *[number_data_received];
  if (this->Ym == nullptr) {
    ERR(L"Can not allocate %zu bytes. At line %d.",
                 static_cast<size_t>(number_data_received) * sizeof(T *),
                 __LINE__);

    this->Deallocate();

    return false;
  }

  tmp_ptr_array_inputs = new T[number_inputs_received * number_data_received];
  if (tmp_ptr_array_inputs == nullptr) {
    ERR(
        L"Can not allocate %zu bytes. At line %d.",
        static_cast<size_t>(number_inputs_received * number_data_received) *
            sizeof(T),
        __LINE__);

    this->Deallocate();

    return false;
  }

  tmp_ptr_array_outputs = new T[number_outputs_received * number_data_received];
  if (tmp_ptr_array_outputs == nullptr) {
    ERR(
        L"Can not allocate %zu bytes. At line %d.",
        static_cast<size_t>(number_outputs_received * number_data_received) *
            sizeof(T),
        __LINE__);

    this->Deallocate();

    return false;
  }

  // Memcpy array inputs.
  struct dim3 tmp_dim3_grid, tmp_dim3_block;

  ptr_Class_Device_Information_received->Grid_Block_1Dimensions(
      number_inputs_received * number_data_received, 0, tmp_dim3_grid,
      tmp_dim3_block);

  Memory::Memory_Copy_1D<T>(number_inputs_received * number_data_received,
                            tmp_ptr_array_inputs, ptr_array_inputs_received,
                            &tmp_dim3_grid, &tmp_dim3_block);
  // |END| Memcpy array inputs. |END|

  // Memcpy array outputs.
  ptr_Class_Device_Information_received->Grid_Block_1Dimensions(
      number_outputs_received * number_data_received, 0, tmp_dim3_grid,
      tmp_dim3_block);

  Memory::Memory_Copy_1D<T>(number_outputs_received * number_data_received,
                            tmp_ptr_array_outputs, ptr_array_outputs_received,
                            &tmp_dim3_grid, &tmp_dim3_block);
  // |END| Memcpy array outputs. |END|

  ptr_Class_Device_Information_received->Grid_Block_1Dimensions(
      number_data_received, 0, *this->ptr_array_dim3_grid_batch,
      *this->ptr_array_dim3_block_batch);

  Two_Memory_Assign_1D_to_2D<T>(
      number_data_received, number_inputs_received, number_outputs_received,
      this->Xm, this->Ym,
      tmp_ptr_array_inputs, tmp_ptr_array_outputs,
      this->ptr_array_dim3_grid_batch, this->ptr_array_dim3_block_batch);

  return true;
}

template <typename T>
__device__ bool cuDataset<T>::valide_spec(
    size_t const &ref_number_inputs_received,
    size_t const &ref_number_outputs_received) const {
  if (ref_number_inputs_received != this->get_n_inp()) {
    ERR(L"Inputs not equal. %d != %d.", ref_number_inputs_received,
                 this->get_n_inp());

    return false;
  } else if (ref_number_outputs_received != this->get_n_out()) {
    ERR(L"Outputs not equal. %d != %d.", ref_number_inputs_received,
                 this->get_n_inp());

    return false;
  } else {
    return true;
  }
}

template <typename T>
__device__ class cuDevicesProp *
cuDataset<T>::Get__Class_Device_Information_Array(void) const {
  return (this->p_ptr_Class_Device_Information_Array);
}

template <typename T>
__global__ void kernel__Dataset_device__Deallocate(
    class cuDataset<T> *const ptr_Dataset_device_received) {
  ptr_Dataset_device_received->Deallocate();
}
template __global__ void kernel__Dataset_device__Deallocate(
    class cuDataset<var> *const);

template <typename T>
__host__ __device__ bool cuDataset<T>::Deallocate(void) {
#ifndef COMPILE_CUDA
  kernel__Dataset_device__Deallocate<T><<<1, 1u>>>(this);

  CUDA__Check_Error();

  return true;
#else
  if (this->_reference == false) {
    if (this->Xm != nullptr) {
      SAFE_DELETE_ARRAY(this->Xm[0]);

      delete[](this->Xm);
      this->Xm = nullptr;
    }

    if (this->Ym != nullptr) {
      SAFE_DELETE_ARRAY(this->Ym[0]);

      delete[](this->Ym);
      this->Ym = nullptr;
    }

    SAFE_DELETE(this->p_ptr_Class_Device_Information_Array);

    // cuRAND.
    if (this->ptr_array_cuRAND_State_MTGP32_shuffle != nullptr) {
      SAFE_DELETE_ARRAY(this->ptr_array_cuRAND_State_MTGP32_shuffle->k);

      delete (this->ptr_array_cuRAND_State_MTGP32_shuffle);
    }
    // |END| cuRAND. |END|
  }

  SAFE_FREE(this->ptr_array_dim3_grid_batch);
  SAFE_FREE(this->ptr_array_dim3_block_batch);

  SAFE_FREE(this->ptr_array_dim3_grid_batch_fold);
  SAFE_FREE(this->ptr_array_dim3_block_batch_fold);

  SAFE_FREE(this->ptr_array_dim3_grid_shuffle);
  SAFE_FREE(this->ptr_array_dim3_block_shuffle);

  // Mini-Batch Stochastic
  SAFE_DELETE_ARRAY(this->ptr_array_stochastic_index);

  SAFE_DELETE_ARRAY(this->ptr_array_inputs_array_stochastic);
  SAFE_DELETE_ARRAY(this->ptr_array_outputs_array_stochastic);
  // - Mini-Batch Stochastic -

  // Cross Validation k-fold
  SAFE_DELETE_ARRAY(this->ptr_array_inputs_array_k_fold);
  SAFE_DELETE_ARRAY(this->ptr_array_outputs_array_k_fold);
  // - Cross Validation k-fold -

  this->p_type_dataset_process =
      DL::DATASET::NONE;

  return true;
#endif
}

template <typename T>
__host__ __device__ bool cuDataset<T>::Get__Use__Shuffle(void) const {
  return (this->use_shuffle);
}

template <typename T>
__global__ void kernel__Dataset_device__Get__Total_Data(
    size_t *const ptr_number_data_received,
    class cuDataset<T> const *const ptr_Dataset_device_received) {
  *ptr_number_data_received = ptr_Dataset_device_received->Get__Total_Data();
}

template <typename T>
__host__ __device__ size_t cuDataset<T>::Get__Total_Data(void) const {
#ifndef COMPILE_CUDA
  size_t n_data, *tmp_ptr_device_number_data;

  CUDA__Safe_Call(
      cudaMalloc((void **)&tmp_ptr_device_number_data, sizeof(size_t)));

  kernel__Dataset_device__Get__Total_Data<T>
      <<<1, 1u>>>(tmp_ptr_device_number_data, this);

  CUDA__Check_Error();

  CUDA__Safe_Call(cudaMemcpy(&n_data, tmp_ptr_device_number_data,
                             sizeof(size_t), cudaMemcpyDeviceToHost));

  CUDA__Safe_Call(cudaFree(tmp_ptr_device_number_data));

  return (n_data);
#else
  return (this->p_n_data);
#endif
}

template <typename T>
__global__ void kernel__Dataset_device__Get__Number_Data(
    size_t *const ptr_number_data_received,
    class cuDataset<T> const *const ptr_Dataset_device_received) {
  *ptr_number_data_received =
      ptr_Dataset_device_received->get_n_data();
}

template <typename T>
__host__ __device__ size_t cuDataset<T>::get_n_data(void) const {
#ifndef COMPILE_CUDA
  size_t n_data, *tmp_ptr_device_number_data;

  CUDA__Safe_Call(
      cudaMalloc((void **)&tmp_ptr_device_number_data, sizeof(size_t)));

  kernel__Dataset_device__Get__Number_Data<T>
      <<<1, 1u>>>(tmp_ptr_device_number_data, this);

  CUDA__Check_Error();

  CUDA__Safe_Call(cudaMemcpy(&n_data, tmp_ptr_device_number_data,
                             sizeof(size_t), cudaMemcpyDeviceToHost));

  CUDA__Safe_Call(cudaFree(tmp_ptr_device_number_data));

  return (n_data);
#else
  switch (this->Get__Type_Dataset_Process()) {
    case DL::DATASET::
        MINIBATCH:
      return (this->p_number_data_mini_batch);
    case DL::DATASET::
        CROSS_VAL:
      return (this->number_data_k_fold);
    default:
      return (this->p_n_data);
  }
#endif
}

template <typename T>
__host__ __device__ size_t cuDataset<T>::Get__Number_CV_K_Fold(void) const {
  return (this->number_k_fold);
}

template <typename T>
__host__ __device__ size_t cuDataset<T>::Get__Number_CV_K_Sub_Fold(void) const {
  return (this->number_k_sub_fold);
}

template <typename T>
__host__ __device__ size_t
cuDataset<T>::Get__Number_CV_Data_Per_Fold(void) const {
  return (this->number_data_per_fold);
}

template <typename T>
__host__ __device__ size_t
cuDataset<T>::Get__Number_CV_Data_Training(void) const {
  return (this->number_data_training);
}

template <typename T>
__host__ __device__ size_t
cuDataset<T>::Get__Number_CV_Data_Validating(void) const {
  return (this->number_data_validating);
}

template <typename T>
__host__ __device__ size_t
cuDataset<T>::Get__Number_CV_Data_Per_Sub_Iteration(void) const {
  return (this->number_data_per_sub_iteration);
}

template <typename T>
__host__ __device__ size_t
cuDataset<T>::Get__Number_CV_Data_Last_Sub_Iteration(void) const {
  return (this->number_data_last_sub_iteration);
}

template <typename T>
__host__ __device__ size_t cuDataset<T>::get_n_inp(void) const {
  return (this->p_n_inp);
}

template <typename T>
__host__ __device__ size_t cuDataset<T>::get_n_out(void) const {
  return (this->p_n_out);
}

template <typename T>
__host__ __device__ size_t
cuDataset<T>::get_seq_w(void) const {
  return (this->p_seq_w);
}

template <typename T>
__global__ void kernel__Dataset_device__Training_Process_Batch(
    var *const ptr_loss_received, var *const ptr_accuracy_received,
    class cuModel *const ptr_cuModel_received,
    class cuDataset<T> *const ptr_Dataset_device_received) {
  ptr_Dataset_device_received->device__Training_Process_Batch(
      *ptr_loss_received, *ptr_accuracy_received, ptr_cuModel_received);
}

template <typename T>
__host__ var cuDataset<T>::Training_Process_Batch(
    class Model *const model) {
  var tmp_loss, tmp_accuracy, *tmp_ptr_device_loss, *tmp_ptr_device_accuracy;

  CUDA__Safe_Call(cudaMalloc((void **)&tmp_ptr_device_loss, sizeof(var)));
  CUDA__Safe_Call(cudaMalloc((void **)&tmp_ptr_device_accuracy, sizeof(var)));

  kernel__Dataset_device__Training_Process_Batch<T><<<1, 1u>>>(
      tmp_ptr_device_loss, tmp_ptr_device_accuracy,
      model->cumodel, this);

  CUDA__Check_Error();

  CUDA__Safe_Call(cudaMemcpy(&tmp_loss, tmp_ptr_device_loss, sizeof(var),
                             cudaMemcpyDeviceToHost));
  CUDA__Safe_Call(cudaMemcpy(&tmp_accuracy, tmp_ptr_device_accuracy, sizeof(var),
                             cudaMemcpyDeviceToHost));

  CUDA__Safe_Call(cudaFree(tmp_ptr_device_loss));      // var
  CUDA__Safe_Call(cudaFree(tmp_ptr_device_accuracy));  // var

  model->is_update_from_device = false;

  model->set_loss(
      DL::ENV::TRAIN, tmp_loss);
  model->set_accu(
      DL::ENV::TRAIN, tmp_accuracy);

  return (model->get_loss(
      DL::ENV::TRAIN));
}

template <typename T>
__device__ void cuDataset<T>::device__Training_Process_Batch(
    var &ref_loss_received, var &ref_accuracy_received,
    class cuModel *const ptr_cuModel_received) {
  if (ptr_cuModel_received->update_mem_thread_size(
          this->get_n_data()) == false) {
    ERR(
        L"An error has been triggered from the "
        "\"update_mem_thread_size(%u)\" function. At line %d.", this->get_n_data(), __LINE__);

    return;
  }

  if (ptr_cuModel_received->update_mem_batch_size(
          this->get_n_data()) == false) {
    ERR(
        L"An error has been triggered from the "
        "\"update_mem_batch_size(%u)\" function. At line %d.", this->get_n_data(), __LINE__);

    return;
  }

  this->Train_Epoch_Batch(ptr_cuModel_received);

  ptr_cuModel_received->Update_Parameter(this->get_n_data(),
                                                 this->Get__Total_Data());

  ++ptr_cuModel_received->epoch_time_step;

  ref_loss_received = ptr_cuModel_received->get_loss(
      DL::ENV::NONE);
  ref_accuracy_received = ptr_cuModel_received->get_accu(
      DL::ENV::NONE);

  ptr_cuModel_received->set_loss(
      DL::ENV::TRAIN,
      ref_loss_received);
  ptr_cuModel_received->set_accu(
      DL::ENV::TRAIN,
      ref_accuracy_received);
}

template <typename T>
__global__ void kernel__Dataset_device__Training_Process_Mini_Batch_Stochastic(
    var *const ptr_loss_received, var *const ptr_accuracy_received,
    class cuModel *const ptr_cuModel_received,
    class cuDataset<T> *const ptr_Dataset_device_received) {
  ptr_Dataset_device_received->device__Training_Process_Mini_Batch_Stochastic(
      *ptr_loss_received, *ptr_accuracy_received, ptr_cuModel_received);
}

template <typename T>
__host__ var cuDataset<T>::Training_Process_Mini_Batch_Stochastic(
    class Model *const model) {
  var tmp_loss, tmp_accuracy, *tmp_ptr_device_loss, *tmp_ptr_device_accuracy;

  CUDA__Safe_Call(cudaMalloc((void **)&tmp_ptr_device_loss, sizeof(var)));
  CUDA__Safe_Call(cudaMalloc((void **)&tmp_ptr_device_accuracy, sizeof(var)));

  kernel__Dataset_device__Training_Process_Mini_Batch_Stochastic<T><<<1, 1u>>>(
      tmp_ptr_device_loss, tmp_ptr_device_accuracy,
      model->cumodel, this);

  CUDA__Check_Error();

  CUDA__Safe_Call(cudaMemcpy(&tmp_loss, tmp_ptr_device_loss, sizeof(var),
                             cudaMemcpyDeviceToHost));
  CUDA__Safe_Call(cudaMemcpy(&tmp_accuracy, tmp_ptr_device_accuracy, sizeof(var),
                             cudaMemcpyDeviceToHost));

  CUDA__Safe_Call(cudaFree(tmp_ptr_device_loss));      // var
  CUDA__Safe_Call(cudaFree(tmp_ptr_device_accuracy));  // var

  model->is_update_from_device = false;

  model->set_loss(
      DL::ENV::TRAIN, tmp_loss);
  model->set_accu(
      DL::ENV::TRAIN, tmp_accuracy);

  return (model->get_loss(
      DL::ENV::TRAIN));
}

template <typename T>
__device__ void cuDataset<T>::device__Training_Process_Mini_Batch_Stochastic(
    var &ref_loss_received, var &ref_accuracy_received,
    class cuModel *const ptr_cuModel_received) {
  if (ptr_cuModel_received->update_mem_thread_size(
          this->p_number_data_mini_batch) == false) {
    ERR(
        L"An error has been triggered from the "
        "\"update_mem_thread_size(%u)\" function. At line %d.", this->p_number_data_mini_batch, __LINE__);

    return;
  }

  if (ptr_cuModel_received->update_mem_batch_size(
          this->p_number_data_mini_batch) == false) {
    ERR(
        L"An error has been triggered from the "
        "\"update_mem_batch_size(%u)\" function. At line %d.", this->p_number_data_mini_batch, __LINE__);

    return;
  }

  var tmp_summation_loss(0_r), tmp_summation_accurancy(0_r);

  if (this->use_shuffle) {
    this->Mini_Batch_Stochastic__Shuffle();
  }

  for (size_t j(0u); j != this->p_number_mini_batch; ++j) {
    if (this->Mini_Batch_Stochastic__Increment_Mini_Batch(j)) {
      this->Train_Epoch_Mini_Batch_Stochastic(ptr_cuModel_received);

      tmp_summation_loss += ptr_cuModel_received->get_loss(
          DL::ENV::NONE);
      tmp_summation_accurancy += ptr_cuModel_received->get_accu(
          DL::ENV::NONE);

      ptr_cuModel_received->Update_Parameter(
          this->p_number_data_mini_batch, this->Get__Total_Data());
    } else {
      ERR(
          L"From "
          "\"Mini_Batch_Stochastic__Increment_Mini_Batch\".",);

      return;
    }
  }

  this->Mini_Batch_Stochastic__Reset();

  ++ptr_cuModel_received->epoch_time_step;

  ref_loss_received = tmp_summation_loss /=
      static_cast<var>(this->p_number_mini_batch);
  ref_accuracy_received = tmp_summation_accurancy /=
      static_cast<var>(this->p_number_mini_batch);

  ptr_cuModel_received->set_loss(
      DL::ENV::TRAIN,
      tmp_summation_loss);
  ptr_cuModel_received->set_accu(
      DL::ENV::TRAIN,
      tmp_summation_accurancy);
}

template <typename T>
__device__ void cuDataset<T>::Mini_Batch_Stochastic__Initialize_Shuffle(void) {
  class cuDeviceProp const *const tmp_ptr_CUDA_Device(
      this->Get__Class_Device_Information_Array()->Get__CUDA_Device());

  // Tree shift shuffle.
  if (this->ptr_array_dim3_grid_shuffle == NULL) {
    struct dim3 *tmp_ptr_array_dim3_grid_shuffle(
        static_cast<struct dim3 *>(malloc(sizeof(struct dim3))));
    if (tmp_ptr_array_dim3_grid_shuffle == NULL) {
      ERR(
          L"Can not allocate memory. malloc(sizeof(%u))", sizeof(struct dim3));

      return;
    }
    *tmp_ptr_array_dim3_grid_shuffle = dim3(1, 1, 1u);
    this->ptr_array_dim3_grid_shuffle = tmp_ptr_array_dim3_grid_shuffle;
  }

  if (this->ptr_array_dim3_block_shuffle == NULL) {
    struct dim3 *tmp_ptr_array_dim3_block_shuffle(
        static_cast<struct dim3 *>(malloc(sizeof(struct dim3))));
    if (tmp_ptr_array_dim3_block_shuffle == NULL) {
      ERR(
          L"Can not allocate memory. malloc(sizeof(%u))", sizeof(struct dim3));

      return;
    }
    *tmp_ptr_array_dim3_block_shuffle = dim3(1, 1, 1u);
    this->ptr_array_dim3_block_shuffle = tmp_ptr_array_dim3_block_shuffle;
  }

  this->p_number_blocks_shuffle = static_cast<size_t>(
      ceil(static_cast<double>(this->p_n_data) /
           static_cast<double>(this->Get__Class_Device_Information_Array()
                                   ->Get__CUDA_Device()
                                   ->Get__Warp_Size())));

  tmp_ptr_CUDA_Device->Grid_Block_cuRAND_1Dimensions(
      this->p_number_blocks_shuffle, 0, this->ptr_array_dim3_grid_shuffle[0],
      this->ptr_array_dim3_block_shuffle[0]);
  // |END| Tree shift shuffle. |END|
}

template <typename T>
__device__ void cuDataset<T>::Mini_Batch_Stochastic__Shuffle(void) {
  Memory::Memory_Initialize_Index_Shift<size_t>(
      this->p_n_data,
      curand(this->ptr_array_cuRAND_State_MTGP32_shuffle) %
          this->p_n_data,
      this->ptr_array_stochastic_index, this->ptr_array_dim3_grid_batch,
      this->ptr_array_dim3_block_batch);

  Shuffle::Tree_Shuffle<size_t>(
      this->p_number_blocks_shuffle,
      this->Get__Class_Device_Information_Array()
          ->Get__CUDA_Device()
          ->Get__Warp_Size(),
      this->p_n_data, this->ptr_array_stochastic_index,
      this->ptr_array_cuRAND_State_MTGP32_shuffle,
      this->ptr_array_dim3_grid_shuffle, this->ptr_array_dim3_block_shuffle);
}

template <typename T>
__global__ void
kernel__Dataset_device__Training_Process_Cross_Validation_K_Fold(
    var *const ptr_loss_received, var *const ptr_accuracy_received,
    class cuModel *const ptr_cuModel_received,
    class cuDataset<T> *const ptr_Dataset_device_received) {
  ptr_Dataset_device_received->device__Training_Process_Cross_Validation_K_Fold(
      *ptr_loss_received, *ptr_accuracy_received, ptr_cuModel_received);
}

template <typename T>
__host__ var cuDataset<T>::Training_Process_Cross_Validation_K_Fold(
    class Model *const model) {
  var tmp_loss, tmp_accuracy, *tmp_ptr_device_loss, *tmp_ptr_device_accuracy;

  CUDA__Safe_Call(cudaMalloc((void **)&tmp_ptr_device_loss, sizeof(var)));
  CUDA__Safe_Call(cudaMalloc((void **)&tmp_ptr_device_accuracy, sizeof(var)));

  kernel__Dataset_device__Training_Process_Cross_Validation_K_Fold<T>
      <<<1, 1u>>>(tmp_ptr_device_loss, tmp_ptr_device_accuracy,
                   model->cumodel,
                   this);

  CUDA__Check_Error();

  CUDA__Safe_Call(cudaMemcpy(&tmp_loss, tmp_ptr_device_loss, sizeof(var),
                             cudaMemcpyDeviceToHost));
  CUDA__Safe_Call(cudaMemcpy(&tmp_accuracy, tmp_ptr_device_accuracy, sizeof(var),
                             cudaMemcpyDeviceToHost));

  CUDA__Safe_Call(cudaFree(tmp_ptr_device_loss));      // var
  CUDA__Safe_Call(cudaFree(tmp_ptr_device_accuracy));  // var

  model->is_update_from_device = false;

  model->set_loss(
      DL::ENV::TRAIN, tmp_loss);
  model->set_accu(
      DL::ENV::TRAIN, tmp_accuracy);

  return (model->get_loss(
      DL::ENV::TRAIN));
}

template <typename T>
__device__ void cuDataset<T>::device__Training_Process_Cross_Validation_K_Fold(
    var &ref_loss_received, var &ref_accuracy_received,
    class cuModel *const ptr_cuModel_received) {
  // Training.
  if (ptr_cuModel_received->update_mem_thread_size(
          this->number_data_k_fold) == false) {
    ERR(
        L"An error has been triggered from the "
        "\"update_mem_thread_size(%u)\" function. At line %d.", this->number_data_k_fold, __LINE__);

    return;
  }

  if (ptr_cuModel_received->update_mem_batch_size(
          this->number_data_k_fold) == false) {
    ERR(
        L"An error has been triggered from the "
        "\"update_mem_batch_size(%u)\" function. At line %d.", this->number_data_k_fold, __LINE__);

    return;
  }

  // Validation.
  if (ptr_cuModel_received->update_mem_thread_size(
          this->ptr_Validation_Dataset->number_data_k_fold) == false) {
    ERR(
        L"An error has been triggered from the "
        "\"update_mem_thread_size(%u)\" function. At line %d.", this->ptr_Validation_Dataset->number_data_k_fold,
        __LINE__);

    return;
  }

  var tmp_summation_loss(0_r), tmp_summation_accurancy(0_r);

  if (this->use_shuffle) {
    this->Cross_Validation_K_Fold__Shuffle();
  }

  for (size_t j(0u), k; j != this->number_k_fold; ++j) {
    if (this->Cross_Validation_K_Fold__Increment_Fold(j)) {
      for (k = 0u; k != this->number_k_sub_fold; ++k) {
        if (this->Cross_Validation_K_Fold__Increment_Sub_Fold(k)) {
          this->Train_Epoch_Cross_Validation_K_Fold(
              ptr_cuModel_received);

          ptr_cuModel_received->Update_Parameter(
              this->number_data_k_fold, this->Get__Total_Data());
        } else {
          ERR(
              L"From "
              "\"Cross_Validation_K_Fold__Increment_Sub_Fold\".",);

          return;
        }
      }

      tmp_summation_loss +=
          this->ptr_Validation_Dataset->Test_Epoch_Cross_Validation_K_Fold(
              ptr_cuModel_received);
      tmp_summation_accurancy += ptr_cuModel_received->get_accu(
          DL::ENV::NONE);
    } else {
      ERR(
          L"From "
          "\"Cross_Validation_K_Fold__Increment_Fold\".",);

      return;
    }
  }

  this->Cross_Validation_K_Fold__Reset();

  ++ptr_cuModel_received->epoch_time_step;

  ref_loss_received = tmp_summation_loss /=
      static_cast<var>(this->number_k_fold);
  ref_accuracy_received = tmp_summation_accurancy /=
      static_cast<var>(this->number_k_fold);

  ptr_cuModel_received->set_loss(
      DL::ENV::TRAIN,
      tmp_summation_loss);
  ptr_cuModel_received->set_accu(
      DL::ENV::TRAIN,
      tmp_summation_accurancy);
}

template <typename T>
__global__ void kernel__Dataset_device__Testing(
    var *const ptr_loss_received, var *const ptr_accuracy_received,
    class cuModel *const ptr_cuModel_received,
    class cuDataset<T> *const ptr_Dataset_device_received) {
  ptr_Dataset_device_received->device__Testing(
      *ptr_loss_received, *ptr_accuracy_received, ptr_cuModel_received);
}

template <typename T>
__device__ void cuDataset<T>::device__Testing(
    var &ref_loss_received, var &ref_accuracy_received,
    class cuModel *const ptr_cuModel_received) {
  if (this->valide_spec(ptr_cuModel_received->n_inp,
                           ptr_cuModel_received->n_out) ==
      false) {
    ERR(
        L"An error has been triggered from the \"valide_spec(%u, "
        "%u)\" function. At line %d.", ptr_cuModel_received->n_inp,
        ptr_cuModel_received->n_out, __LINE__);

    ref_loss_received = 1.0f;

    return;
  }

  ptr_cuModel_received->reset_loss();

  size_t const n_data(this->Get__Total_Data()),
      tmp_maximum_batch_size(ptr_cuModel_received->batch_size),
      tmp_number_batchs(static_cast<size_t>(
          ceil(static_cast<double>(n_data) /
               static_cast<double>(tmp_maximum_batch_size))));
  size_t tmp_batch_size, i;

  if (ptr_cuModel_received->update_mem_thread_size(n_data) ==
      false) {
    ERR(
        L"An error has been triggered from the "
        "\"update_mem_thread_size(%u)\" function. At line %d.", n_data, __LINE__);

    return;
  }

  for (i = 0u; i != tmp_number_batchs; ++i) {
    tmp_batch_size = i + 1u != tmp_number_batchs
                         ? tmp_maximum_batch_size
                         : n_data - i * tmp_maximum_batch_size;

    ptr_cuModel_received->forward_pass(
        tmp_batch_size, this->Get__Input_Array() + i * tmp_maximum_batch_size);

    ptr_cuModel_received->Test(
        tmp_batch_size, this->Get__Output_Array() + i * tmp_maximum_batch_size);
  }

  *ptr_cuModel_received->ptr_array_number_loss =
      n_data * this->get_n_out();
  ptr_cuModel_received->n_acc_trial =
      n_data * this->get_n_out();

  // Synchronize the computed error before merging between threads.
  CUDA__Check_Error();

  ptr_cuModel_received->merge_mp_accu_loss();

  ref_loss_received = ptr_cuModel_received->get_loss(
      DL::ENV::NONE);
  ref_accuracy_received = ptr_cuModel_received->get_accu(
      DL::ENV::NONE);

  ptr_cuModel_received->set_loss(
      DL::ENV::TESTG, ref_loss_received);
  ptr_cuModel_received->set_accu(
      DL::ENV::TESTG,
      ref_accuracy_received);
}

template <typename T>
__host__ var
cuDataset<T>::evaluate(class Model *const model) {
  var tmp_loss, tmp_accuracy, *tmp_ptr_device_loss, *tmp_ptr_device_accuracy;

  CUDA__Safe_Call(cudaMalloc((void **)&tmp_ptr_device_loss, sizeof(var)));
  CUDA__Safe_Call(cudaMalloc((void **)&tmp_ptr_device_accuracy, sizeof(var)));

  kernel__Dataset_device__Testing<T><<<1, 1u>>>(
      tmp_ptr_device_loss, tmp_ptr_device_accuracy,
      model->cumodel, this);

  CUDA__Check_Error();

  CUDA__Safe_Call(cudaMemcpy(&tmp_loss, tmp_ptr_device_loss, sizeof(var),
                             cudaMemcpyDeviceToHost));
  CUDA__Safe_Call(cudaMemcpy(&tmp_accuracy, tmp_ptr_device_accuracy, sizeof(var),
                             cudaMemcpyDeviceToHost));

  CUDA__Safe_Call(cudaFree(tmp_ptr_device_loss));      // var
  CUDA__Safe_Call(cudaFree(tmp_ptr_device_accuracy));  // var

  model->set_loss(
      DL::ENV::TESTG, tmp_loss);
  model->set_accu(
      DL::ENV::TESTG, tmp_accuracy);

  return (model->get_loss(
      DL::ENV::TESTG));
}

template <typename T>
__host__ __device__ DL::DATASET
cuDataset<T>::Get__Type_Dataset_Process(void) const {
  return (this->p_type_dataset_process);
}

template <typename T>
__device__ T cuDataset<T>::get_inp(
    size_t const index_received, size_t const sub_index_received) const {
  return (this->Xm[index_received][sub_index_received]);
}

template <typename T>
__device__ T cuDataset<T>::get_out(
    size_t const index_received, size_t const sub_index_received) const {
  return (this->Ym[index_received][sub_index_received]);
}

template <typename T>
__device__ T *cuDataset<T>::get_inp(size_t const index_received) const {
  return (this->Xm[index_received]);
}

template <typename T>
__device__ T *cuDataset<T>::get_out(size_t const index_received) const {
  return (this->Ym[index_received]);
}

template <typename T>
__device__ T **cuDataset<T>::Get__Input_Array(void) const {
  return (this->Xm);
}

template <typename T>
__device__ T **cuDataset<T>::Get__Output_Array(void) const {
  return (this->Ym);
}

template <typename T>
__global__ void kernel__Dataset_device__Get__Sizeof(
    size_t *const ptr_size_t_received,
    class cuDataset<T> const *const ptr_Dataset_device_received) {
  *ptr_size_t_received = ptr_Dataset_device_received->Get__Sizeof();
}
template __global__ void kernel__Dataset_device__Get__Sizeof(
    size_t *const ptr_size_t_received, class cuDataset<var> const *const);

template <typename T>
__host__ __device__ size_t cuDataset<T>::Get__Sizeof(void) const {
  size_t tmp_total_size_t(0_UZ);

#ifndef COMPILE_CUDA
  size_t *tmp_ptr_device_total_size_t;

  CUDA__Safe_Call(
      cudaMalloc((void **)&tmp_ptr_device_total_size_t, sizeof(size_t)));

  kernel__Dataset_device__Get__Sizeof<T>
      <<<1, 1u>>>(tmp_ptr_device_total_size_t, this);

  CUDA__Check_Error();

  CUDA__Safe_Call(cudaMemcpy(&tmp_total_size_t, tmp_ptr_device_total_size_t,
                             sizeof(size_t), cudaMemcpyDeviceToHost));

  CUDA__Safe_Call(cudaFree(tmp_ptr_device_total_size_t));

  return (tmp_total_size_t);
#else
  tmp_total_size_t += sizeof(class cuDataset<T>);  // this

  if (this->_reference == false && this->Xm != nullptr) {
    tmp_total_size_t += this->p_n_data * sizeof(T *);
    tmp_total_size_t +=
        this->p_n_data * this->p_n_inp * sizeof(T);
  }

  if (this->_reference == false && this->Ym != nullptr) {
    tmp_total_size_t += this->p_n_data * sizeof(T *);
    tmp_total_size_t +=
        this->p_n_data * this->p_n_out * sizeof(T);
  }

  if (this->ptr_array_dim3_grid_batch != NULL) {
    tmp_total_size_t += sizeof(struct dim3);
  }
  if (this->ptr_array_dim3_block_batch != NULL) {
    tmp_total_size_t += sizeof(struct dim3);
  }

  if (this->ptr_array_dim3_grid_batch_fold != NULL) {
    tmp_total_size_t += sizeof(struct dim3);
  }
  if (this->ptr_array_dim3_block_batch_fold != NULL) {
    tmp_total_size_t += sizeof(struct dim3);
  }

  if (this->ptr_array_dim3_grid_shuffle != NULL) {
    tmp_total_size_t += sizeof(struct dim3);
  }
  if (this->ptr_array_dim3_block_shuffle != NULL) {
    tmp_total_size_t += sizeof(struct dim3);
  }

  // Mini-Batch Stochastic
  if (this->ptr_array_stochastic_index != nullptr) {
    tmp_total_size_t += this->p_n_data * sizeof(size_t);
  }

  if (this->ptr_array_inputs_array_stochastic != nullptr) {
    tmp_total_size_t += this->p_number_data_last_iteration * sizeof(T *);
  }
  if (this->ptr_array_outputs_array_stochastic != nullptr) {
    tmp_total_size_t += this->p_number_data_last_iteration * sizeof(T *);
  }
  // - Mini-Batch Stochastic -

  // Cross Validation k-fold
  // TODO: Sizeof training || Sizeof validating
  if (this->ptr_array_inputs_array_k_fold != nullptr) {
    tmp_total_size_t += this->number_data_training * sizeof(T *);
  }
  if (this->ptr_array_outputs_array_k_fold != nullptr) {
    tmp_total_size_t += this->number_data_training * sizeof(T *);
  }
  // - Cross Validation k-fold -

  // cuRAND.
  if (this->ptr_array_cuRAND_State_MTGP32_shuffle != nullptr) {
    tmp_total_size_t += this->p_number_cuRAND_State_MTGP32_shuffle *
                        sizeof(struct curandStateMtgp32);
    tmp_total_size_t += this->p_number_cuRAND_State_MTGP32_shuffle *
                        sizeof(struct mtgp32_kernel_params);
  }
  // |END| cuRAND. |END|

  return (tmp_total_size_t);
#endif
}

// template initialization declaration.
template class cuDataset<var>;
