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

#include "deep-learning-lib/v1/ops/distributions/shuffle.cuh"
#include "deep-learning-lib/v1/ops/distributions/curand.cuh"
#include "deep-learning-lib/v1/learner/model.hpp"
#include "deep-learning-lib/v1/learner/model.cuh"
#include "deep-learning-lib/v1/data/datasets.cuh"
#include "deep-learning-lib/io/file.hpp"
#include "deep-learning-lib/io/logger.hpp"

#include <curand_kernel.h>

#include <chrono>

template <typename T>
__host__ __device__ cuDatasets<T>::cuDatasets(void) {}

template <typename T>
__host__ __device__ cuDatasets<T>::~cuDatasets(void) {
  this->Deallocate();
}

template <typename T>
__global__ void kernel__CUDA_Dataset_Manager__Copy(
    size_t const number_data_received, size_t const number_inputs_received,
    size_t const number_outputs_received,
    size_t const number_recurrent_depth_received,
    T *const ptr_array_inputs_received, T *const ptr_array_outputs_received,
    class cuDeviceProp *const ptr_Class_Device_Information_received,
    class cuDatasets<T> *const ptr_CUDA_Dataset_Manager_received) {
  ptr_CUDA_Dataset_Manager_received->device_Copy(
      number_data_received, number_inputs_received, number_outputs_received,
      number_recurrent_depth_received, ptr_array_inputs_received,
      ptr_array_outputs_received, ptr_Class_Device_Information_received);
}

template <typename T>
__host__ bool cuDatasets<T>::copy(
    class Datasets *const datasets) {
  size_t const n_data(
      datasets->get_n_data()),
      tmp_number_inputs(datasets->get_n_inp()),
      tmp_number_outputs(datasets->get_n_out()),
      tmp_number_time_predictions(
          datasets->get_seq_w());
  int device_id(0);

  T *tmp_ptr_device_array_inputs_array, *tmp_ptr_device_array_outputs_array;

  class cuDeviceProp *tmp_ptr_Class_Device_Information;

  CUDA__Safe_Call(cudaMalloc((void **)&tmp_ptr_Class_Device_Information,
                             sizeof(class cuDeviceProp)));

  CUDA__Safe_Call(cudaGetDevice(&device_id));

  tmp_ptr_Class_Device_Information->Initialize(device_id);

  CUDA__Safe_Call(
      cudaMalloc((void **)&tmp_ptr_device_array_inputs_array,
                 tmp_number_inputs * n_data * sizeof(T)));
  CUDA__Safe_Call(
      cudaMalloc((void **)&tmp_ptr_device_array_outputs_array,
                 tmp_number_outputs * n_data * sizeof(T)));

  CUDA__Safe_Call(
      cudaMemcpy(tmp_ptr_device_array_inputs_array,
                 datasets->Get__Input_Array(),
                 tmp_number_inputs * n_data * sizeof(T),
                 cudaMemcpyHostToDevice));
  CUDA__Safe_Call(
      cudaMemcpy(tmp_ptr_device_array_outputs_array,
                 datasets->Get__Output_Array(),
                 tmp_number_outputs * n_data * sizeof(T),
                 cudaMemcpyHostToDevice));

  kernel__CUDA_Dataset_Manager__Copy<T>
      <<<1, 1u>>>(n_data,                 // size_t
                   tmp_number_inputs,                   // size_t
                   tmp_number_outputs,                  // size_t
                   tmp_number_time_predictions,         // size_t
                   tmp_ptr_device_array_inputs_array,   // T
                   tmp_ptr_device_array_outputs_array,  // T
                   tmp_ptr_Class_Device_Information,    // class cuDeviceProp
                   this);                               // class

  CUDA__Check_Error();

  CUDA__Safe_Call(
      cudaFree(tmp_ptr_Class_Device_Information));  // class cuDeviceProp
  CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_inputs_array));   // T
  CUDA__Safe_Call(cudaFree(tmp_ptr_device_array_outputs_array));  // T

  if (this->Initialize_CUDA_Device() == false) {
    ERR(
        L"An error has been triggered from the "
        "\"Initialize_CUDA_Device()\" function.",);

    return false;
  } else if (this->Initialize_cuRAND(static_cast<unsigned int>(
                 std::chrono::high_resolution_clock::now()
                     .time_since_epoch()
                     .count())) == false) {
    ERR(
        L"An error has been triggered from the "
        "\"Initialize_cuRAND(random)\" function.",);

    return false;
  }

  return true;
}

template <typename T>
__device__ bool cuDatasets<T>::device_Copy(
    size_t const number_data_received, size_t const number_inputs_received,
    size_t const number_outputs_received,
    size_t const number_recurrent_depth_received,
    T const *ptr_array_inputs_received, T const *ptr_array_outputs_received,
    class cuDeviceProp *const ptr_Class_Device_Information_received) {
  T *tmp_ptr_array_inputs, *tmp_ptr_array_outputs;

  this->p_n_data = number_data_received;

  this->p_n_inp = number_inputs_received;
  this->p_n_out = number_outputs_received;
  this->p_seq_w = number_recurrent_depth_received;

  this->Xm = new T *[number_data_received];
  if (this->Xm == nullptr) {
    ERR(L"Can not allocate %zu bytes.",
                 static_cast<size_t>(number_data_received) * sizeof(T *));

    this->Deallocate();

    return false;
  }

  this->Ym = new T *[number_data_received];
  if (this->Ym == nullptr) {
    ERR(L"Can not allocate %zu bytes.",
                 static_cast<size_t>(number_data_received) * sizeof(T *));

    this->Deallocate();

    return false;
  }

  tmp_ptr_array_inputs = new T[number_inputs_received * number_data_received];
  if (tmp_ptr_array_inputs == nullptr) {
    ERR(
        L"Can not allocate %zu bytes.",
        static_cast<size_t>(number_inputs_received * number_data_received) *
            sizeof(T));

    this->Deallocate();

    return false;
  }

  tmp_ptr_array_outputs = new T[number_outputs_received * number_data_received];
  if (tmp_ptr_array_outputs == nullptr) {
    ERR(
        L"Can not allocate %zu bytes.",
        static_cast<size_t>(number_outputs_received * number_data_received) *
            sizeof(T));

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
      number_data_received, 0, tmp_dim3_grid, tmp_dim3_block);

  Two_Memory_Assign_1D_to_2D<T>(
      number_data_received, number_inputs_received, number_outputs_received,
      this->Xm, this->Ym,
      tmp_ptr_array_inputs, tmp_ptr_array_outputs, &tmp_dim3_grid,
      &tmp_dim3_block);

  return true;
}

template <typename T>
__global__ void kernel__CUDA_Dataset_Manager__Deallocate(
    class cuDatasets<T> *const ptr_CUDA_Dataset_Manager_received) {
  ptr_CUDA_Dataset_Manager_received->Deallocate();
}
template __global__ void kernel__CUDA_Dataset_Manager__Deallocate(
    class cuDatasets<var> *const);

template <typename T>
__host__ __device__ bool cuDatasets<T>::Deallocate(void) {
#ifndef COMPILE_CUDA
  kernel__CUDA_Dataset_Manager__Deallocate<T><<<1, 1u>>>(this);

  CUDA__Check_Error();

  return true;
#else
  SAFE_DELETE(this->p_ptr_Class_Device_Information_Array);

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

  if (this->_ptr_array_Dataset != nullptr) {
    switch (this->_type_storage_data) {
      case ENUM_TYPE_DATASET_MANAGER_STORAGE::
          TYPE_STORAGE_TRAINING:
        if (this->_ptr_array_Dataset[0].Deallocate() == false) {
          ERR(
              L"An error has been triggered from the "
              "\"[0].Deallocate()\" function.",);

          return false;
        }
        break;
      case ENUM_TYPE_DATASET_MANAGER_STORAGE::
          TYPE_STORAGE_TRAINING_TESTING:
        if (this->_ptr_array_Dataset[0].Deallocate() == false) {
          ERR(
              L"An error has been triggered from the "
              "\"[0].Deallocate()\" function.",);

          return false;
        }

        if (this->_ptr_array_Dataset[1].Deallocate() == false) {
          ERR(
              L"An error has been triggered from the "
              "\"[1].Deallocate()\" function.",);

          return false;
        }
        break;
      case ENUM_TYPE_DATASET_MANAGER_STORAGE::
          TYPE_STORAGE_TRAINING_VALIDATION_TESTING:
        if (this->_ptr_array_Dataset[0].Deallocate() == false) {
          ERR(
              L"An error has been triggered from the "
              "\"[0].Deallocate()\" function.",);

          return false;
        }

        if (this->_ptr_array_Dataset[1].Deallocate() == false) {
          ERR(
              L"An error has been triggered from the "
              "\"[1].Deallocate()\" function.",);

          return false;
        }

        if (this->_ptr_array_Dataset[2].Deallocate() == false) {
          ERR(
              L"An error has been triggered from the "
              "\"[2].Deallocate()\" function.",);

          return false;
        }
        break;
      default:
        ERR(
            L"DatasetV1 storage type (%d) is not managed in the "
            "switch.", this->_type_storage_data);
        break;
    }

    delete[](this->_ptr_array_Dataset);
    this->_ptr_array_Dataset = nullptr;
  }

  // cuRAND.
  if (this->ptr_array_cuRAND_State_MTGP32_shuffle != nullptr) {
    SAFE_DELETE_ARRAY(this->ptr_array_cuRAND_State_MTGP32_shuffle->k);

    delete (this->ptr_array_cuRAND_State_MTGP32_shuffle);
  }
  // |END| cuRAND. |END|

  return true;
#endif
}

template <typename T>
__global__ void kernel__CUDA_Dataset_Manager__Initialize(
    class cuDatasets<T> *const ptr_CUDA_Dataset_Manager_received) {
  ptr_CUDA_Dataset_Manager_received->Initialize();
}
template __global__ void kernel__CUDA_Dataset_Manager__Initialize(
    class cuDatasets<var> *const);

template <typename T>
__device__ class cuDevicesProp *
cuDatasets<T>::Get__Class_Device_Information_Array(void) const {
  return (this->p_ptr_Class_Device_Information_Array);
}

template <typename T>
__global__ void kernel__CUDA_Dataset_Manager__Add_CUDA_Device(
    int const index_device_received,
    struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received,
    class cuDatasets<T> *const ptr_Dataset_device_received) {
  ptr_Dataset_device_received->Add_CUDA_Device(
      index_device_received, ptr_struct_cudaDeviceProp_received);
}

template <typename T>
__device__ bool cuDatasets<T>::Add_CUDA_Device(
    int const index_device_received,
    struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received) {
  if (this->p_ptr_Class_Device_Information_Array == nullptr) {
    this->p_ptr_Class_Device_Information_Array = new class cuDevicesProp;
  }

  return (this->p_ptr_Class_Device_Information_Array->push_back(
      index_device_received, ptr_struct_cudaDeviceProp_received));
}

template <typename T>
__host__ bool cuDatasets<T>::Initialize_CUDA_Device(void) {
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

    kernel__CUDA_Dataset_Manager__Add_CUDA_Device<<<1, 1u>>>(
        device_id, tmp_ptr_device_struct_cudaDeviceProp, this);

    CUDA__Check_Error();
  }

  CUDA__Safe_Call(cudaFree(tmp_ptr_device_struct_cudaDeviceProp));

  return true;
}

template <typename T>
__host__ __device__ bool cuDatasets<T>::Initialize(void) {
#ifndef COMPILE_CUDA
  kernel__CUDA_Dataset_Manager__Initialize<T><<<1, 1u>>>(this);

  CUDA__Check_Error();

  return true;
#else
  this->p_n_data = 0u;
  this->p_seq_w = 0u;
  this->p_n_inp = 0u;
  this->p_n_out = 0u;

  this->Xm = nullptr;
  this->Ym = nullptr;

  this->_type_storage_data =
      ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_NONE;

  this->p_ptr_Class_Device_Information_Array = nullptr;

  this->_ptr_array_Dataset = nullptr;

  // cuRAND.
  this->p_number_cuRAND_State_MTGP32_shuffle = 0u;

  this->ptr_array_cuRAND_State_MTGP32_shuffle = nullptr;
  // |END| cuRAND. |END|

  return true;
#endif
}

template <typename T>
__global__ void kernel__CUDA_Dataset_Manager__Initialize(
    DL::ENV::TYPE const env_type,
    DL::DATASET::TYPE const
        type_gradient_descent_received,
    class cuDatasets<T> *const ptr_CUDA_Dataset_Manager_received) {
  ptr_CUDA_Dataset_Manager_received->Initialize(env_type,
                                                type_gradient_descent_received);
}
template __global__ void kernel__CUDA_Dataset_Manager__Initialize(
    DL::ENV::TYPE const env_type,
    DL::DATASET::TYPE const,
    class cuDatasets<var> *const);

template <typename T>
__host__ __device__ bool cuDatasets<T>::Initialize(
    DL::ENV::TYPE const env_type,
    DL::DATASET::TYPE const type_gradient_descent_received) {
#ifndef COMPILE_CUDA
  kernel__CUDA_Dataset_Manager__Initialize<T>
      <<<1, 1u>>>(env_type, type_gradient_descent_received, this);

  CUDA__Check_Error();

  return true;
#else
  class cuDataset<T> *const tmp_ptr_Dataset_device(
      this->get_dataset(env_type));

  if (tmp_ptr_Dataset_device == nullptr) {
    ERR(
        L"An error has been triggered from the "
        "\"get_dataset(%u)\" function.", env_type);

    return false;
  } else if (tmp_ptr_Dataset_device->Initialize(
                 type_gradient_descent_received) == false) {
    ERR(
        L"An error has been triggered from the \"Initialize(%u)\" "
        "function.", type_gradient_descent_received);

    return false;
  }

  return true;
#endif
}

template <typename T>
__global__ void
kernel__CUDA_Dataset_Manager__Initialize_Mini_Batch_Stochastic_Gradient_Descent(
    bool const use_shuffle_received,
    size_t const desired_number_data_per_mini_batch_received,
    size_t const number_mini_batch_maximum_received,
    class cuDatasets<T> *const ptr_CUDA_Dataset_Manager_received) {
  ptr_CUDA_Dataset_Manager_received
      ->Initialize_Mini_Batch_Stochastic_Gradient_Descent(
          use_shuffle_received, desired_number_data_per_mini_batch_received,
          number_mini_batch_maximum_received);
}
template __global__ void
kernel__CUDA_Dataset_Manager__Initialize_Mini_Batch_Stochastic_Gradient_Descent(
    bool const, size_t const, size_t const, class cuDatasets<var> *const);

template <typename T>
__host__ __device__ bool
cuDatasets<T>::Initialize_Mini_Batch_Stochastic_Gradient_Descent(
    bool const use_shuffle_received,
    size_t const desired_number_data_per_mini_batch_received,
    size_t const number_mini_batch_maximum_received) {
#ifndef COMPILE_CUDA
  kernel__CUDA_Dataset_Manager__Initialize_Mini_Batch_Stochastic_Gradient_Descent<
      T><<<1, 1u>>>(use_shuffle_received,
                     desired_number_data_per_mini_batch_received,
                     number_mini_batch_maximum_received, this);

  CUDA__Check_Error();

  return true;
#else
  class cuDataset<T> *tmp_ptr_Dataset_device(this->get_dataset(
      DL::ENV::TRAIN));

  if (tmp_ptr_Dataset_device == nullptr) {
    ERR(
        L"An error has been triggered from the "
        "\"get_dataset(%u)\" function.", DL::ENV::TRAIN);

    return false;
  } else if (tmp_ptr_Dataset_device->Get__Type_Dataset_Process() !=
             DL::DATASET::
                 MINIBATCH) {
    ERR(
        L"The dataset process (%u) differs from the mini-batch "
        "process (%u).", tmp_ptr_Dataset_device->Get__Type_Dataset_Process(),
        DL::DATASET::
            MINIBATCH);

    return false;
  }

  tmp_ptr_Dataset_device->Initialize_Mini_Batch_Stochastic_Gradient_Descent(
      use_shuffle_received, desired_number_data_per_mini_batch_received,
      number_mini_batch_maximum_received);

  return true;
#endif
}

template <typename T>
__global__ void
kernel__CUDA_Dataset_Manager__Initialize_Cross_Validation_K_Fold(
    bool const use_shuffle_received, size_t const number_k_fold_received,
    size_t const number_k_sub_fold_received,
    class cuDatasets<T> *const ptr_CUDA_Dataset_Manager_received) {
  ptr_CUDA_Dataset_Manager_received->Initialize__Cross_Validation(
      use_shuffle_received, number_k_fold_received, number_k_sub_fold_received);
}
template __global__ void
kernel__CUDA_Dataset_Manager__Initialize_Cross_Validation_K_Fold(
    bool const, size_t const, size_t const, class cuDatasets<var> *const);

template <typename T>
__host__ __device__ bool cuDatasets<T>::Initialize__Cross_Validation(
    bool const use_shuffle_received, size_t const number_k_fold_received,
    size_t const number_k_sub_fold_received) {
#ifndef COMPILE_CUDA
  kernel__CUDA_Dataset_Manager__Initialize_Cross_Validation_K_Fold<T>
      <<<1, 1u>>>(use_shuffle_received, number_k_fold_received,
                   number_k_sub_fold_received, this);

  CUDA__Check_Error();

  return true;
#else
  class cuDataset<T> *tmp_ptr_Dataset_device(this->get_dataset(
      DL::ENV::TRAIN));

  if (tmp_ptr_Dataset_device == nullptr) {
    ERR(
        L"An error has been triggered from the "
        "\"get_dataset(%u)\" function.", DL::ENV::TRAIN);

    return false;
  } else if (number_k_fold_received < 2u) {
    ERR(L"Not enough K-fold.");

    return false;
  } else if (tmp_ptr_Dataset_device->Get__Type_Dataset_Process() !=
             DL::DATASET::
                 CROSS_VAL) {
    ERR(
        L"The dataset process (%u) differs from the cross validating "
        "k-fold process (%u).", tmp_ptr_Dataset_device->Get__Type_Dataset_Process(),
        DL::DATASET::
            CROSS_VAL);

    return false;
  }

  tmp_ptr_Dataset_device->Initialize__Cross_Validation(
      use_shuffle_received, number_k_fold_received, number_k_sub_fold_received,
      this);

  return true;
#endif
}

template <typename T>
__global__ void
kernel__CUDA_Dataset_Manager__Initialize_Cross_Validation_K_Fold(
    class cuDatasets<T> *const ptr_CUDA_Dataset_Manager_received) {
  ptr_CUDA_Dataset_Manager_received->Initialize__Cross_Validation();
}
template __global__ void
kernel__CUDA_Dataset_Manager__Initialize_Cross_Validation_K_Fold(
    class cuDatasets<var> *const);

template <typename T>
__host__ __device__ bool cuDatasets<T>::Initialize__Cross_Validation(void) {
#ifndef COMPILE_CUDA
  kernel__CUDA_Dataset_Manager__Initialize_Cross_Validation_K_Fold<T>
      <<<1, 1u>>>(this);

  CUDA__Check_Error();

  return true;
#else
  class cuDataset<T> *tmp_ptr_Dataset_device(this->get_dataset(
      DL::ENV::VALID));

  if (tmp_ptr_Dataset_device == nullptr) {
    ERR(
        L"An error has been triggered from the "
        "\"get_dataset(%u)\" function.", DL::ENV::VALID);

    return false;
  } else if (tmp_ptr_Dataset_device->Get__Type_Dataset_Process() !=
             DL::DATASET::
                 CROSS_VAL) {
    ERR(
        L"The dataset process (%u) differs from the cross validating "
        "k-fold process (%u).", tmp_ptr_Dataset_device->Get__Type_Dataset_Process(),
        DL::DATASET::
            CROSS_VAL);

    return false;
  }

  tmp_ptr_Dataset_device->Initialize__Cross_Validation(this);

  return true;
#endif
}

template <typename T>
__global__ void kernel__CUDA_Dataset_Manager__Set__Type_Gradient_Descent(
    DL::ENV::TYPE const env_type,
    DL::DATASET::TYPE const
        type_gradient_descent_received,
    class cuDatasets<T> *const ptr_CUDA_Dataset_Manager_received) {
  ptr_CUDA_Dataset_Manager_received->Set__Type_Gradient_Descent(
      env_type, type_gradient_descent_received);
}
template __global__ void
kernel__CUDA_Dataset_Manager__Set__Type_Gradient_Descent(
    DL::ENV::TYPE const,
    DL::DATASET::TYPE const,
    class cuDatasets<var> *const);

template <typename T>
__host__ __device__ bool cuDatasets<T>::Set__Type_Gradient_Descent(
    DL::ENV::TYPE const env_type,
    DL::DATASET::TYPE const type_gradient_descent_received) {
#ifndef COMPILE_CUDA
  kernel__CUDA_Dataset_Manager__Set__Type_Gradient_Descent<T>
      <<<1, 1u>>>(env_type, type_gradient_descent_received, this);

  CUDA__Check_Error();

  return true;
#else
  class cuDataset<T> *const tmp_ptr_Dataset_device(
      this->get_dataset(env_type));

  if (tmp_ptr_Dataset_device == nullptr) {
    ERR(
        L"An error has been triggered from the "
        "\"get_dataset(%u)\" function.", env_type);

    return false;
  } else if (tmp_ptr_Dataset_device->Set__Type_Gradient_Descent(
                 type_gradient_descent_received) == false) {
    ERR(
        L"An error has been triggered from the "
        "\"Set__Type_Gradient_Descent(%u)\" function.", type_gradient_descent_received);

    return false;
  }

  return true;
#endif
}

template <typename T>
__device__ T Get__Limit(T const value_received, T const minimum_received,
                        T const maximum_received) {
  if (value_received < minimum_received) {
    return (minimum_received);
  } else if (value_received > maximum_received) {
    return (maximum_received);
  } else {
    return (value_received);
  }
}

template <typename T>
__device__ T Get__Minimum(T const value_received, T const minimum_received) {
  if (value_received < minimum_received) {
    return (minimum_received);
  } else {
    return (value_received);
  }
}

template <typename T>
__device__ T Get__Maximum(T const value_received, T const maximum_received) {
  if (value_received > maximum_received) {
    return (maximum_received);
  } else {
    return (value_received);
  }
}

template <typename T>
__global__ void kernel__CUDA_Dataset_Manager__Prepare_Storage(
    class cuDatasets<T> *const ptr_CUDA_Dataset_Manager_received) {
  ptr_CUDA_Dataset_Manager_received->Prepare_Storage();
}
template __global__ void kernel__CUDA_Dataset_Manager__Prepare_Storage(
    class cuDatasets<var> *const);

template <typename T>
__host__ __device__ bool cuDatasets<T>::Prepare_Storage(void) {
#ifndef COMPILE_CUDA
  kernel__CUDA_Dataset_Manager__Prepare_Storage<T><<<1, 1u>>>(this);

  CUDA__Check_Error();

  return true;
#else
  if (this->get_n_data() == 0u) {
    ERR(L"Number of data equal to zero.");

    return false;
  } else if (this->_type_storage_data !=
             ENUM_TYPE_DATASET_MANAGER_STORAGE::
                 TYPE_STORAGE_NONE) {
    ERR(L"Can not prepare storage multiple time.");

    return false;
  }

  this->_ptr_array_Dataset = new class cuDataset<T>[1];

  this->_ptr_array_Dataset[0].reference(
      this->get_n_data(), this->p_n_inp,
      this->p_n_out, this->p_seq_w,
      this->Xm, this->Ym,
      this->p_number_cuRAND_State_MTGP32_shuffle,
      this->ptr_array_cuRAND_State_MTGP32_shuffle,
      this->p_ptr_Class_Device_Information_Array);

  this->_type_storage_data =
      ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING;

  return true;
#endif
}

template <typename T>
__global__ void kernel__Dataset_Manager__Prepare_Storage(
    size_t const number_data_training_received,
    size_t const number_data_testing_received,
    class cuDatasets<T> *const datasets) {
  datasets->Prepare_Storage(number_data_training_received,
                                                number_data_testing_received);
}
template __global__ void kernel__Dataset_Manager__Prepare_Storage(
    size_t const, size_t const, class cuDatasets<var> *const);

template <typename T>
__host__ __device__ bool cuDatasets<T>::Prepare_Storage(
    size_t const number_data_training_received,
    size_t const number_data_testing_received) {
  if (number_data_training_received == 0u) {
    ERR(L"Number of training data equal to zero.");

    return false;
  } else if (number_data_testing_received == 0u) {
    ERR(L"Number of testing data equal to zero.");

    return false;
  }

#ifndef COMPILE_CUDA
  kernel__Dataset_Manager__Prepare_Storage<T><<<1, 1u>>>(
      number_data_training_received, number_data_testing_received, this);

  CUDA__Check_Error();

  return true;
#else
  if (number_data_training_received + number_data_testing_received !=
      this->get_n_data()) {
    ERR(L"training(%d) + testing(%d) != data(%d)", number_data_training_received,
                 number_data_testing_received, this->get_n_data());

    return false;
  } else if (this->get_n_data() < 2u) {
    ERR(L"Number of data (%u) < 2",
                 this->get_n_data());

    return false;
  } else if (this->_type_storage_data !=
             ENUM_TYPE_DATASET_MANAGER_STORAGE::
                 TYPE_STORAGE_NONE) {
    ERR(L"Can not prepare storage multiple time.",);

    return false;
  }

  T **tmp_ptr_array_inputs_array(this->Xm),
      **tmp_ptr_array_outputs_array(this->Ym);

  this->_ptr_array_Dataset = new class cuDataset<T>[2];

  this->_ptr_array_Dataset[0].reference(
      number_data_training_received, this->p_n_inp,
      this->p_n_out, this->p_seq_w,
      tmp_ptr_array_inputs_array, tmp_ptr_array_outputs_array,
      this->p_number_cuRAND_State_MTGP32_shuffle,
      this->ptr_array_cuRAND_State_MTGP32_shuffle,
      this->p_ptr_Class_Device_Information_Array);

  tmp_ptr_array_inputs_array += number_data_training_received;
  tmp_ptr_array_outputs_array += number_data_training_received;

  this->_ptr_array_Dataset[1].reference(
      number_data_testing_received, this->p_n_inp,
      this->p_n_out, this->p_seq_w,
      tmp_ptr_array_inputs_array, tmp_ptr_array_outputs_array,
      this->p_number_cuRAND_State_MTGP32_shuffle,
      this->ptr_array_cuRAND_State_MTGP32_shuffle,
      this->p_ptr_Class_Device_Information_Array);

  this->_type_storage_data = ENUM_TYPE_DATASET_MANAGER_STORAGE::
      TYPE_STORAGE_TRAINING_TESTING;

  return true;
#endif
}

template <typename T>
__global__ void kernel__Dataset_Manager__Prepare_Storage(
    size_t const number_data_training_received,
    size_t const number_data_validation_received,
    size_t const number_data_testing_received,
    class cuDatasets<T> *const datasets) {
  datasets->Prepare_Storage(number_data_training_received,
                                                number_data_validation_received,
                                                number_data_testing_received);
}
template __global__ void kernel__Dataset_Manager__Prepare_Storage(
    size_t const, size_t const, size_t const, class cuDatasets<var> *const);

template <typename T>
__host__ __device__ bool cuDatasets<T>::Prepare_Storage(
    size_t const number_data_training_received,
    size_t const number_data_validation_received,
    size_t const number_data_testing_received) {
  if (number_data_training_received == 0u) {
    ERR(L"Number of training data equal to zero.",);

    return false;
  } else if (number_data_validation_received == 0u) {
    ERR(L"Number of validating data equal to zero.",);

    return false;
  } else if (number_data_testing_received == 0u) {
    ERR(L"Number of testing data equal to zero.",);

    return false;
  }

#ifndef COMPILE_CUDA
  kernel__Dataset_Manager__Prepare_Storage<T><<<1, 1u>>>(
      number_data_training_received, number_data_validation_received,
      number_data_testing_received, this);

  CUDA__Check_Error();

  return true;
#else
  if (number_data_training_received + number_data_validation_received +
          number_data_testing_received !=
      this->get_n_data()) {
    ERR(
        L"training(%d) + validation(%d) + testing(%d) != "
        "data(%d)", number_data_training_received,
        number_data_validation_received, number_data_testing_received,
        this->get_n_data());

    return false;
  } else if (this->get_n_data() < 3u) {
    ERR(L"Number of data (%u) < 3",
                 this->get_n_data());

    return false;
  } else if (this->_type_storage_data !=
             ENUM_TYPE_DATASET_MANAGER_STORAGE::
                 TYPE_STORAGE_NONE) {
    ERR(L"Can not prepare storage multiple time.",);

    return false;
  }

  T **tmp_ptr_array_inputs_array(this->Xm),
      **tmp_ptr_array_outputs_array(this->Ym);

  this->_ptr_array_Dataset = new class cuDataset<T>[3];

  this->_ptr_array_Dataset[0].reference(
      number_data_training_received, this->p_n_inp,
      this->p_n_out, this->p_seq_w,
      tmp_ptr_array_inputs_array, tmp_ptr_array_outputs_array,
      this->p_number_cuRAND_State_MTGP32_shuffle,
      this->ptr_array_cuRAND_State_MTGP32_shuffle,
      this->p_ptr_Class_Device_Information_Array);

  tmp_ptr_array_inputs_array += number_data_training_received;
  tmp_ptr_array_outputs_array += number_data_training_received;

  this->_ptr_array_Dataset[1].reference(
      number_data_validation_received, this->p_n_inp,
      this->p_n_out, this->p_seq_w,
      tmp_ptr_array_inputs_array, tmp_ptr_array_outputs_array,
      this->p_number_cuRAND_State_MTGP32_shuffle,
      this->ptr_array_cuRAND_State_MTGP32_shuffle,
      this->p_ptr_Class_Device_Information_Array);

  tmp_ptr_array_inputs_array += number_data_validation_received;
  tmp_ptr_array_outputs_array += number_data_validation_received;

  this->_ptr_array_Dataset[2].reference(
      number_data_testing_received, this->p_n_inp,
      this->p_n_out, this->p_seq_w,
      tmp_ptr_array_inputs_array, tmp_ptr_array_outputs_array,
      this->p_number_cuRAND_State_MTGP32_shuffle,
      this->ptr_array_cuRAND_State_MTGP32_shuffle,
      this->p_ptr_Class_Device_Information_Array);

  this->_type_storage_data = ENUM_TYPE_DATASET_MANAGER_STORAGE::
      TYPE_STORAGE_TRAINING_VALIDATION_TESTING;

  return true;
#endif
}

template <typename T>
__global__ void kernel__Dataset_Manager__Prepare_Storage(
    var const number_data_percent_training_received,
    var const number_data_percent_testing_received,
    class cuDatasets<T> *const datasets) {
  datasets->Prepare_Storage(
      number_data_percent_training_received,
      number_data_percent_testing_received);
}
template __global__ void kernel__Dataset_Manager__Prepare_Storage(
    var const, var const, class cuDatasets<var> *const);

template <typename T>
__host__ __device__ bool cuDatasets<T>::Prepare_Storage(
    var const number_data_percent_training_received,
    var const number_data_percent_testing_received) {
  if (number_data_percent_training_received +
          number_data_percent_testing_received !=
      100_r) {
    ERR(
        L"training(%f%%) + testing(%f%%) != 100.0%%", number_data_percent_training_received,
        number_data_percent_testing_received);

    return false;
  } else if (number_data_percent_training_received == 0_r) {
    ERR(L"training(%f%%) == 0.0%%",
                 number_data_percent_training_received);

    return false;
  } else if (number_data_percent_testing_received == 0_r) {
    ERR(L"testing(%f%%) == 0.0%%",
                 number_data_percent_testing_received);

    return false;
  }

#ifndef COMPILE_CUDA
  kernel__Dataset_Manager__Prepare_Storage<T>
      <<<1, 1u>>>(number_data_percent_training_received,
                   number_data_percent_testing_received, this);

  CUDA__Check_Error();

  return true;
#else
  if (this->get_n_data() < 2u) {
    ERR(L"Number of data (%u) < 2",
                 this->get_n_data());

    return false;
  } else if (this->_type_storage_data !=
             ENUM_TYPE_DATASET_MANAGER_STORAGE::
                 TYPE_STORAGE_NONE) {
    ERR(L"Can not prepare storage multiple time.",);

    return false;
  }

  size_t const tmp_number_data_training(Get__Minimum<size_t>(
      static_cast<size_t>(
          round(static_cast<double>(this->get_n_data()) *
                number_data_percent_training_received / 100.0)),
      1u)),
      tmp_number_data_testing(this->get_n_data() -
                              tmp_number_data_training);

  T **tmp_ptr_array_inputs_array(this->Xm),
      **tmp_ptr_array_outputs_array(this->Ym);

  this->_ptr_array_Dataset = new class cuDataset<T>[2];

  this->_ptr_array_Dataset[0].reference(
      tmp_number_data_training, this->p_n_inp, this->p_n_out,
      this->p_seq_w, tmp_ptr_array_inputs_array,
      tmp_ptr_array_outputs_array, this->p_number_cuRAND_State_MTGP32_shuffle,
      this->ptr_array_cuRAND_State_MTGP32_shuffle,
      this->p_ptr_Class_Device_Information_Array);

  tmp_ptr_array_inputs_array += tmp_number_data_training;
  tmp_ptr_array_outputs_array += tmp_number_data_training;

  this->_ptr_array_Dataset[1].reference(
      tmp_number_data_testing, this->p_n_inp, this->p_n_out,
      this->p_seq_w, tmp_ptr_array_inputs_array,
      tmp_ptr_array_outputs_array, this->p_number_cuRAND_State_MTGP32_shuffle,
      this->ptr_array_cuRAND_State_MTGP32_shuffle,
      this->p_ptr_Class_Device_Information_Array);

  this->_type_storage_data = ENUM_TYPE_DATASET_MANAGER_STORAGE::
      TYPE_STORAGE_TRAINING_TESTING;

  return true;
#endif
}

template <typename T>
__global__ void kernel__Dataset_Manager__Prepare_Storage(
    var const number_data_percent_training_received,
    var const number_data_percent_validation_received,
    var const number_data_percent_testing_received,
    class cuDatasets<T> *const datasets) {
  datasets->Prepare_Storage(
      number_data_percent_training_received,
      number_data_percent_validation_received,
      number_data_percent_testing_received);
}
template __global__ void kernel__Dataset_Manager__Prepare_Storage(
    var const, var const, var const, class cuDatasets<var> *const);

template <typename T>
__host__ __device__ bool cuDatasets<T>::Prepare_Storage(
    var const number_data_percent_training_received,
    var const number_data_percent_validation_received,
    var const number_data_percent_testing_received) {
  if (number_data_percent_training_received +
          number_data_percent_validation_received +
          number_data_percent_testing_received !=
      100_r) {
    ERR(
        L"training(%f%%) + validation(%f%%) + testing(%f%%) != "
        "100.0%%", number_data_percent_training_received,
        number_data_percent_validation_received,
        number_data_percent_testing_received);

    return false;
  } else if (number_data_percent_training_received == 0_r) {
    ERR(L"training(%f%%) == 0.0%%",
                 number_data_percent_training_received);

    return false;
  } else if (number_data_percent_validation_received == 0_r) {
    ERR(L"validation(%f%%) == 0.0%%",
                 number_data_percent_validation_received);

    return false;
  } else if (number_data_percent_testing_received == 0_r) {
    ERR(L"testing(%f%%) == 0.0%%",
                 number_data_percent_testing_received);

    return false;
  }

#ifndef COMPILE_CUDA
  kernel__Dataset_Manager__Prepare_Storage<T>
      <<<1, 1u>>>(number_data_percent_training_received,
                   number_data_percent_validation_received,
                   number_data_percent_testing_received, this);

  CUDA__Check_Error();

  return true;
#else
  if (this->get_n_data() < 3u) {
    ERR(L"Number of data (%u) < 3",
                 this->get_n_data());

    return false;
  } else if (this->_type_storage_data !=
             ENUM_TYPE_DATASET_MANAGER_STORAGE::
                 TYPE_STORAGE_NONE) {
    ERR(L"Can not prepare storage multiple time.",);

    return false;
  }

  size_t const tmp_number_data_training(Get__Limit<size_t>(
      static_cast<size_t>(
          round(static_cast<double>(this->get_n_data()) *
                number_data_percent_training_received / 100.0)),
      1, this->get_n_data() - 2u)),
      tmp_number_data_validation(Get__Limit<size_t>(
          static_cast<size_t>(
              round(static_cast<double>(this->get_n_data()) *
                    number_data_percent_validation_received / 100.0)),
          1, this->get_n_data() - tmp_number_data_training - 1u)),
      tmp_number_data_testing(Get__Minimum<size_t>(
          this->get_n_data() - tmp_number_data_training -
              tmp_number_data_validation,
          1u));

  T **tmp_ptr_array_inputs_array(this->Xm),
      **tmp_ptr_array_outputs_array(this->Ym);

  this->_ptr_array_Dataset = new class cuDataset<T>[3];

  this->_ptr_array_Dataset[0].reference(
      tmp_number_data_training, this->p_n_inp, this->p_n_out,
      this->p_seq_w, tmp_ptr_array_inputs_array,
      tmp_ptr_array_outputs_array, this->p_number_cuRAND_State_MTGP32_shuffle,
      this->ptr_array_cuRAND_State_MTGP32_shuffle,
      this->p_ptr_Class_Device_Information_Array);

  tmp_ptr_array_inputs_array += tmp_number_data_training;
  tmp_ptr_array_outputs_array += tmp_number_data_training;

  this->_ptr_array_Dataset[1].reference(
      tmp_number_data_validation, this->p_n_inp, this->p_n_out,
      this->p_seq_w, tmp_ptr_array_inputs_array,
      tmp_ptr_array_outputs_array, this->p_number_cuRAND_State_MTGP32_shuffle,
      this->ptr_array_cuRAND_State_MTGP32_shuffle,
      this->p_ptr_Class_Device_Information_Array);

  tmp_ptr_array_inputs_array += tmp_number_data_validation;
  tmp_ptr_array_outputs_array += tmp_number_data_validation;

  this->_ptr_array_Dataset[2].reference(
      tmp_number_data_testing, this->p_n_inp, this->p_n_out,
      this->p_seq_w, tmp_ptr_array_inputs_array,
      tmp_ptr_array_outputs_array, this->p_number_cuRAND_State_MTGP32_shuffle,
      this->ptr_array_cuRAND_State_MTGP32_shuffle,
      this->p_ptr_Class_Device_Information_Array);

  this->_type_storage_data = ENUM_TYPE_DATASET_MANAGER_STORAGE::
      TYPE_STORAGE_TRAINING_VALIDATION_TESTING;

  return true;
#endif
}

template <typename T>
__global__ void kernel__CUDA_Dataset_Manager__Get__Number_Data(
    size_t *ptr_number_data_received,
    class cuDatasets<T> const *const ptr_CUDA_Dataset_Manager_received) {
  *ptr_number_data_received =
      ptr_CUDA_Dataset_Manager_received->get_n_data();
}

template <typename T>
__host__ __device__ size_t cuDatasets<T>::get_n_data(void) const {
#ifndef COMPILE_CUDA
  size_t n_data, *tmp_ptr_device_number_data;

  CUDA__Safe_Call(
      cudaMalloc((void **)&tmp_ptr_device_number_data, sizeof(size_t)));

  kernel__CUDA_Dataset_Manager__Get__Number_Data<T>
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
__host__ __device__ size_t cuDatasets<T>::get_n_inp(void) const {
  return (this->p_n_inp);
}

template <typename T>
__host__ __device__ size_t cuDatasets<T>::get_n_out(void) const {
  return (this->p_n_out);
}

template <typename T>
__host__ __device__ size_t
cuDatasets<T>::get_seq_w(void) const {
  return (this->p_seq_w);
}

template <typename T>
__device__ void cuDatasets<T>::train(
    var &ref_loss_received, var &ref_accuracy_received,
    class cuModel *const ptr_cuModel_received) {
  class cuDataset<T> *const tmp_ptr_Dataset_device(this->get_dataset(
      DL::ENV::TRAIN));

  if (tmp_ptr_Dataset_device == nullptr) {
    ERR(
        L"An error has been triggered from the "
        "\"get_dataset(%u)\" function.", DL::ENV::TRAIN);

    return;
  } else if (tmp_ptr_Dataset_device->valide_spec(
                 ptr_cuModel_received->n_inp,
                 ptr_cuModel_received->n_out) == false) {
    ERR(
        L"An error has been triggered from the \"valide_spec(%u, "
        "%u)\" function.", ptr_cuModel_received->n_inp,
        ptr_cuModel_received->n_out);

    return;
  }

  ptr_cuModel_received->type_state_propagation = DL::
      PROPAGATION::TRAINING;

  switch (DL::DATASET::BATCH) {
    case DL::DATASET::BATCH:
      tmp_ptr_Dataset_device->device__Training_Process_Batch(
          ref_loss_received, ref_accuracy_received,
          ptr_cuModel_received);
      break;
    case DL::DATASET::
        MINIBATCH:
      tmp_ptr_Dataset_device->device__Training_Process_Mini_Batch_Stochastic(
          ref_loss_received, ref_accuracy_received,
          ptr_cuModel_received);
      break;
    case DL::DATASET::
        CROSS_VAL:
      tmp_ptr_Dataset_device->device__Training_Process_Cross_Validation_K_Fold(
          ref_loss_received, ref_accuracy_received,
          ptr_cuModel_received);
      break;
    default:
      ref_loss_received = 1_r;
      ref_accuracy_received = 0_r;
      break;
  }

  ptr_cuModel_received->type_state_propagation = DL::
      PROPAGATION::INFERENCE;
}

template <typename T>
__global__ void kernel__CUDA_Dataset_Manager__Training(
    var *const ptr_error_received, var *const ptr_accuracy_received,
    class cuModel *const ptr_cuModel_received,
    class cuDatasets<T> *const ptr_CUDA_Dataset_Manager_received) {
  ptr_CUDA_Dataset_Manager_received->train(*ptr_error_received,
                                              *ptr_accuracy_received,
                                              ptr_cuModel_received);
}

template <typename T>
__host__ var
cuDatasets<T>::train(class Model *const model) {
  var tmp_loss, tmp_accuracy, *tmp_ptr_device_loss, *tmp_ptr_device_accuracy;

  CUDA__Safe_Call(cudaMalloc((void **)&tmp_ptr_device_loss, sizeof(var)));
  CUDA__Safe_Call(cudaMalloc((void **)&tmp_ptr_device_accuracy, sizeof(var)));

  kernel__CUDA_Dataset_Manager__Training<T><<<1, 1u>>>(
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
__device__ void cuDatasets<T>::device__Type_Testing(
    var &ref_loss_received, var &ref_accuracy_received,
    DL::ENV::TYPE const env_type,
    class cuModel *const ptr_cuModel_received) {
  class cuDataset<T> *const tmp_ptr_Dataset_device(
      this->get_dataset(env_type));

  if (tmp_ptr_Dataset_device == nullptr) {
    ERR(
        L"An error has been triggered from the "
        "\"get_dataset(%u)\" function.", DL::ENV::TRAIN);

    return;
  }

  var const tmp_previous_loss(ptr_cuModel_received->get_loss(
      DL::ENV::TESTG)),
      tmp_previous_accuracy(ptr_cuModel_received->get_accu(
          DL::ENV::TESTG));

  tmp_ptr_Dataset_device->device__Testing(
      ref_loss_received, ref_accuracy_received, ptr_cuModel_received);

  switch (env_type) {
    case DL::ENV::TRAIN:
      ptr_cuModel_received->set_loss(
          DL::ENV::TRAIN,
          ref_loss_received);
      ptr_cuModel_received->set_accu(
          DL::ENV::TRAIN,
          ref_accuracy_received);
      break;
    case DL::ENV::VALID:
      ptr_cuModel_received->set_loss(
          DL::ENV::VALID,
          ref_loss_received);
      ptr_cuModel_received->set_accu(
          DL::ENV::VALID,
          ref_accuracy_received);
      break;
  }

  // reset testing loss/accuracy.
  if (env_type !=
      DL::ENV::TESTG) {
    ptr_cuModel_received->set_loss(
        DL::ENV::TESTG,
        tmp_previous_loss);
    ptr_cuModel_received->set_accu(
        DL::ENV::TESTG,
        tmp_previous_accuracy);
  }
  // |END| reset testing loss/accuracy. |END|
}

template <typename T>
__global__ void kernel__CUDA_Dataset_Manager__Type_Testing(
    var *const ptr_loss_received, var *const ptr_accuray_received,
    DL::ENV::TYPE const env_type,
    class cuModel *const ptr_cuModel_received,
    class cuDatasets<T> *const ptr_CUDA_Dataset_Manager_received) {
  ptr_CUDA_Dataset_Manager_received->device__Type_Testing(
      *ptr_loss_received, *ptr_accuray_received, env_type,
      ptr_cuModel_received);
}

template <typename T>
__host__ var cuDatasets<T>::Type_Testing(
    DL::ENV::TYPE const env_type,
    class Model *const model) {
  var tmp_loss(0_r), tmp_accuracy(0_r), *tmp_ptr_device_loss,
      *tmp_ptr_device_accuracy;

  CUDA__Safe_Call(cudaMalloc((void **)&tmp_ptr_device_loss, sizeof(var)));
  CUDA__Safe_Call(cudaMalloc((void **)&tmp_ptr_device_accuracy, sizeof(var)));

  kernel__CUDA_Dataset_Manager__Type_Testing<T><<<1, 1u>>>(
      tmp_ptr_device_loss, tmp_ptr_device_accuracy, env_type,
      model->cumodel, this);

  CUDA__Check_Error();

  CUDA__Safe_Call(cudaMemcpy(&tmp_loss, tmp_ptr_device_loss, sizeof(var),
                             cudaMemcpyDeviceToHost));
  CUDA__Safe_Call(cudaMemcpy(&tmp_accuracy, tmp_ptr_device_accuracy, sizeof(var),
                             cudaMemcpyDeviceToHost));

  CUDA__Safe_Call(cudaFree(tmp_ptr_device_loss));      // var
  CUDA__Safe_Call(cudaFree(tmp_ptr_device_accuracy));  // var

  switch (env_type) {
    case DL::ENV::TRAIN:
      model->set_loss(
          DL::ENV::TRAIN, tmp_loss);
      model->set_accu(
          DL::ENV::TRAIN, tmp_accuracy);
      break;
    case DL::ENV::VALID:
      model->set_loss(
          DL::ENV::VALID, tmp_loss);
      model->set_accu(
          DL::ENV::VALID,
          tmp_accuracy);
      break;
    case DL::ENV::TESTG:
      model->set_loss(
          DL::ENV::TESTG, tmp_loss);
      model->set_accu(
          DL::ENV::TESTG, tmp_accuracy);
      break;
    default:
      ERR(L"Undefined data type (%d).", env_type);
      break;
  }

  return tmp_loss;
}

template <typename T>
__host__ __device__ ENUM_TYPE_DATASET_MANAGER_STORAGE
cuDatasets<T>::get_storage_type(void) const {
  return (this->_type_storage_data);
}

template <typename T>
__device__ T cuDatasets<T>::get_inp(
    size_t const index_received, size_t const sub_index_received) const {
  return (this->Xm[index_received][sub_index_received]);
}

template <typename T>
__device__ T cuDatasets<T>::get_out(
    size_t const index_received, size_t const sub_index_received) const {
  return (this->Ym[index_received][sub_index_received]);
}

template <typename T>
__device__ T *cuDatasets<T>::get_inp(size_t const index_received) const {
  return (this->Xm[index_received]);
}

template <typename T>
__device__ T *cuDatasets<T>::get_out(size_t const index_received) const {
  return (this->Ym[index_received]);
}

template <typename T>
__device__ T **cuDatasets<T>::Get__Input_Array(void) const {
  return (this->Xm);
}

template <typename T>
__device__ T **cuDatasets<T>::Get__Output_Array(void) const {
  return (this->Ym);
}

template <typename T>
__global__ void kernel__CUDA_Dataset_Manager__Get__Sizeof(
    size_t *const ptr_size_t_received,
    class cuDatasets<T> const *const ptr_CUDA_Dataset_Manager_received) {
  *ptr_size_t_received = ptr_CUDA_Dataset_Manager_received->Get__Sizeof();
}
template __global__ void kernel__CUDA_Dataset_Manager__Get__Sizeof(
    size_t *const, class cuDatasets<var> const *const);

template <typename T>
__host__ __device__ size_t cuDatasets<T>::Get__Sizeof(void) const {
  size_t tmp_total_size_t(0_UZ);

#ifndef COMPILE_CUDA
  size_t *tmp_ptr_device_total_size_t;

  CUDA__Safe_Call(
      cudaMalloc((void **)&tmp_ptr_device_total_size_t, sizeof(size_t)));

  kernel__CUDA_Dataset_Manager__Get__Sizeof<T>
      <<<1, 1u>>>(tmp_ptr_device_total_size_t, this);

  CUDA__Check_Error();

  CUDA__Safe_Call(cudaMemcpy(&tmp_total_size_t, tmp_ptr_device_total_size_t,
                             sizeof(size_t), cudaMemcpyDeviceToHost));

  CUDA__Safe_Call(cudaFree(tmp_ptr_device_total_size_t));

  return (tmp_total_size_t);
#else
  tmp_total_size_t += sizeof(class cuDatasets<T>);  // this

  if (this->Xm != nullptr) {
    tmp_total_size_t += this->p_n_data * sizeof(T *);
    tmp_total_size_t +=
        this->p_n_data * this->p_n_inp * sizeof(T);
  }

  if (this->Ym != nullptr) {
    tmp_total_size_t += this->p_n_data * sizeof(T *);
    tmp_total_size_t +=
        this->p_n_data * this->p_n_out * sizeof(T);
  }

  // TODO: Create into cuDevicesProp a function returning sizeof called
  // Get__Sizeof().
  if (this->p_ptr_Class_Device_Information_Array != nullptr) {
    tmp_total_size_t += sizeof(class cuDevicesProp);

    if (this->p_ptr_Class_Device_Information_Array
            ->Get__Number_CUDA_Devices() != 0u) {
      tmp_total_size_t +=
          sizeof(class cuDeviceProp);  // _ptr_Class_Device_Information_sum
      tmp_total_size_t +=
          sizeof(class cuDeviceProp);  // _ptr_Class_Device_Information_higher
      tmp_total_size_t +=
          sizeof(class cuDeviceProp);  // _ptr_Class_Device_Information_lower
      tmp_total_size_t +=
          this->p_ptr_Class_Device_Information_Array
              ->Get__Number_CUDA_Devices() *
          sizeof(class cuDeviceProp);  // _ptr_array_Class_Device_Information
    }
  }

  if (this->_ptr_array_Dataset != nullptr) {
    switch (this->_type_storage_data) {
      case ENUM_TYPE_DATASET_MANAGER_STORAGE::
          TYPE_STORAGE_TRAINING:
        tmp_total_size_t += this->_ptr_array_Dataset[0].Get__Sizeof();
        break;
      case ENUM_TYPE_DATASET_MANAGER_STORAGE::
          TYPE_STORAGE_TRAINING_TESTING:
        tmp_total_size_t += this->_ptr_array_Dataset[0].Get__Sizeof();
        tmp_total_size_t += this->_ptr_array_Dataset[1].Get__Sizeof();
        break;
      case ENUM_TYPE_DATASET_MANAGER_STORAGE::
          TYPE_STORAGE_TRAINING_VALIDATION_TESTING:
        tmp_total_size_t += this->_ptr_array_Dataset[0].Get__Sizeof();
        tmp_total_size_t += this->_ptr_array_Dataset[1].Get__Sizeof();
        tmp_total_size_t += this->_ptr_array_Dataset[2].Get__Sizeof();
        break;
      default:
        ERR(L"Undefined storage type (%d).", this->_type_storage_data);
        break;
    }
  }

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

template <typename T>
__device__ class cuDataset<T> *cuDatasets<T>::get_dataset(
    DL::ENV::TYPE const env_type) const {
  if (this->_type_storage_data ==
      ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING) {
    return (&this->_ptr_array_Dataset[0]);
  } else if (this->_type_storage_data ==
             ENUM_TYPE_DATASET_MANAGER_STORAGE::
                 TYPE_STORAGE_TRAINING_TESTING) {
    switch (env_type) {
      case DL::ENV::TRAIN:
        return (&this->_ptr_array_Dataset[0]);
      case DL::ENV::VALID:
        return (&this->_ptr_array_Dataset[0]);
      case DL::ENV::TESTG:
        return (&this->_ptr_array_Dataset[1]);
      default:
        ERR(L"Undefined data type (%d).", env_type);
        return nullptr;
    }
  } else if (this->_type_storage_data ==
             ENUM_TYPE_DATASET_MANAGER_STORAGE::
                 TYPE_STORAGE_TRAINING_VALIDATION_TESTING) {
    switch (env_type) {
      case DL::ENV::TRAIN:
        return (&this->_ptr_array_Dataset[0]);
      case DL::ENV::VALID:
        return (&this->_ptr_array_Dataset[1]);
      case DL::ENV::TESTG:
        return (&this->_ptr_array_Dataset[2]);
      default:
        ERR(L"Undefined data type (%d).", env_type);
        return nullptr;
    }
  }

  return nullptr;
}

template <typename T>
__host__ void cuDatasets<T>::static_Deallocate_CUDA_Dataset_Manager(
    class cuDatasets<var> *&ptr_CUDA_Dataset_Manager_received) {
  INFO(L"GPU: Data: Deallocate.");

  if (ptr_CUDA_Dataset_Manager_received != NULL &&
      ptr_CUDA_Dataset_Manager_received->Deallocate()) {
    INFO(L"GPU: Data: Free.");

    CUDA__Safe_Call(cudaFree(ptr_CUDA_Dataset_Manager_received));

    ptr_CUDA_Dataset_Manager_received = NULL;
  }
}

template <typename T>
__global__ void kernel__Dataset_device__Initialize_cuRAND_MTGP32(
    int const number_states_MTGP32_received,
    struct curandStateMtgp32 *const ptr_curandStateMtgp32_received,
    class cuDataset<T> *const ptr_Dataset_device_received) {
  if (ptr_Dataset_device_received->Initialize_cuRAND_MTGP32(
          number_states_MTGP32_received, ptr_curandStateMtgp32_received) ==
      false) {
    ERR(
        L"An error has been triggered from the "
        "\"Initialize_cuRAND_MTGP32(%d, ptr)\" function.", number_states_MTGP32_received);
  }
}

template <typename T>
__device__ bool cuDataset<T>::Initialize_cuRAND_MTGP32(
    int const number_states_MTGP32_received,
    struct curandStateMtgp32 *const ptr_curandStateMtgp32_received) {
  if (number_states_MTGP32_received == 0) {
    ERR(
        L"Can not initialize cuRAND. Size of the array equal "
        "zero.",);

    return false;
  }

  struct mtgp32_kernel_params *tmp_ptr_array_mtgp32_kernel_params_t;

  // allocate cuRAND State MTGP32 shuffle.
  struct curandStateMtgp32 *tmp_ptr_array_cuRAND_State_MTGP32_shuffle(
      new struct curandStateMtgp32[number_states_MTGP32_received]);
  if (tmp_ptr_array_cuRAND_State_MTGP32_shuffle == nullptr) {
    ERR(L"Can not allocate %zu bytes.",
                 static_cast<size_t>(number_states_MTGP32_received) *
                     sizeof(struct curandStateMtgp32));

    return false;
  }
  this->ptr_array_cuRAND_State_MTGP32_shuffle =
      tmp_ptr_array_cuRAND_State_MTGP32_shuffle;
  // |END| allocate cuRAND State MTGP32 shuffle. |END|

  // copy cuRAND State MTGP32 shuffle.
  Memory::Copy_Loop<struct curandStateMtgp32>(
      ptr_curandStateMtgp32_received,
      ptr_curandStateMtgp32_received + number_states_MTGP32_received,
      this->ptr_array_cuRAND_State_MTGP32_shuffle);
  // |END| copy cuRAND State MTGP32 shuffle. |END|

  // allocate tmp_ptr_array_mtgp32_kernel_params_t.
  tmp_ptr_array_mtgp32_kernel_params_t =
      new struct mtgp32_kernel_params[number_states_MTGP32_received];
  if (tmp_ptr_array_mtgp32_kernel_params_t == nullptr) {
    ERR(L"Can not allocate %zu bytes.",
                 static_cast<size_t>(number_states_MTGP32_received) *
                     sizeof(struct mtgp32_kernel_params));

    return false;
  }
  // |END| allocate tmp_ptr_array_mtgp32_kernel_params_t. |END|

  // Assign cuRAND State MTGP32 shuffle variable.
  struct dim3 tmp_dim3_grid(1, 1, 1u), tmp_dim3_block(1, 1, 1u);

  if (USE_PARALLEL && number_states_MTGP32_received >= warpSize) {
    this->Get__Class_Device_Information_Array()
        ->Get__CUDA_Device()
        ->Grid_Block_1Dimensions(
            static_cast<size_t>(number_states_MTGP32_received), 0_UZ,
            tmp_dim3_grid, tmp_dim3_block);
  }

  cuRAND__Memcpy_cuRAND_State_MTGP32(
      number_states_MTGP32_received, tmp_ptr_array_cuRAND_State_MTGP32_shuffle,
      ptr_curandStateMtgp32_received, tmp_ptr_array_mtgp32_kernel_params_t,
      &tmp_dim3_grid, &tmp_dim3_block);

  this->p_number_cuRAND_State_MTGP32_shuffle = number_states_MTGP32_received;
  // |END| Assign cuRAND State MTGP32 shuffle variable. |END|

  return true;
}

template <typename T>
__global__ void kernel__Dataset_device__Total_Blocks_cuRAND_MTGP32(
    int *const ptr_number_states_MTGP32_received,
    class cuDataset<T> *ptr_Dataset_device_received) {
  double const tmp_number_blocks(
      ceil(static_cast<double>(ptr_Dataset_device_received->Get__Total_Data()) /
           static_cast<double>(ptr_Dataset_device_received
                                   ->Get__Class_Device_Information_Array()
                                   ->Get__CUDA_Device()
                                   ->Get__Warp_Size())));

  if (tmp_number_blocks > (std::numeric_limits<int>::max)()) {
    ERR(
        L"Overflow conversion (%f) to int (%d).", tmp_number_blocks, (std::numeric_limits<int>::max)());
  }

  *ptr_number_states_MTGP32_received =
      static_cast<int>(ceil(tmp_number_blocks / 256.0));
}

template <typename T>
__host__ bool cuDataset<T>::Initialize_cuRAND(size_t const seed) {
  int tmp_number_states_MTGP32, *tmp_ptr_device_number_states_MTGP32(nullptr);

  CUDA__Safe_Call(
      cudaMalloc((void **)&tmp_ptr_device_number_states_MTGP32, sizeof(int)));

  kernel__Dataset_device__Total_Blocks_cuRAND_MTGP32<T>
      <<<1, 1u>>>(tmp_ptr_device_number_states_MTGP32, this);

  CUDA__Check_Error();

  CUDA__Safe_Call(cudaMemcpy(&tmp_number_states_MTGP32,
                             tmp_ptr_device_number_states_MTGP32, sizeof(int),
                             cudaMemcpyKind::cudaMemcpyDeviceToHost));

  if (tmp_number_states_MTGP32 != 0) {
    struct mtgp32_kernel_params *tmp_ptr_mtgp32_kernel_params(NULL);

    struct curandStateMtgp32 *tmp_ptr_curandStateMtgp32_t(NULL);

    if (Allocate_cuRAND_MTGP32(tmp_number_states_MTGP32, seed,
                               tmp_ptr_mtgp32_kernel_params,
                               tmp_ptr_curandStateMtgp32_t) == false) {
      ERR(
          L"An error has been triggered from the "
          "\"Allocate_cuRAND_MTGP32(%d, %zu, ptr, ptr)\" function.", tmp_number_states_MTGP32, seed);

      CUDA__Safe_Call(cudaFree(tmp_ptr_device_number_states_MTGP32));

      return false;
    }

    kernel__Dataset_device__Initialize_cuRAND_MTGP32<<<1, 1u>>>(
        tmp_number_states_MTGP32, tmp_ptr_curandStateMtgp32_t, this);

    CUDA__Check_Error();

    Cleanup_cuRAND_MTGP32(tmp_ptr_mtgp32_kernel_params,
                          tmp_ptr_curandStateMtgp32_t);
  }

  CUDA__Safe_Call(cudaFree(tmp_ptr_device_number_states_MTGP32));

  return true;
}

template <typename T>
__global__ void kernel__CUDA_Dataset_Manager__Initialize_cuRAND_MTGP32(
    int const number_states_MTGP32_received,
    struct curandStateMtgp32 *const ptr_curandStateMtgp32_received,
    class cuDatasets<T> *const ptr_CUDA_Dataset_Manager_received) {
  if (ptr_CUDA_Dataset_Manager_received->Initialize_cuRAND_MTGP32(
          number_states_MTGP32_received, ptr_curandStateMtgp32_received) ==
      false) {
    ERR(
        L"An error has been triggered from the "
        "\"Initialize_cuRAND_MTGP32(%d, ptr)\" function.", number_states_MTGP32_received);
  }
}

template <typename T>
__device__ bool cuDatasets<T>::Initialize_cuRAND_MTGP32(
    int const number_states_MTGP32_received,
    struct curandStateMtgp32 *const ptr_curandStateMtgp32_received) {
  if (number_states_MTGP32_received == 0u) {
    ERR(
        L"Can not initialize cuRAND. Size of the array equal "
        "zero.",);

    return false;
  }

  struct mtgp32_kernel_params *tmp_ptr_array_mtgp32_kernel_params_t;

  // allocate cuRAND State MTGP32 shuffle.
  struct curandStateMtgp32 *tmp_ptr_array_cuRAND_State_MTGP32_shuffle(
      new struct curandStateMtgp32[number_states_MTGP32_received]);
  if (tmp_ptr_array_cuRAND_State_MTGP32_shuffle == nullptr) {
    ERR(L"Can not allocate %zu bytes.",
                 static_cast<size_t>(number_states_MTGP32_received) *
                     sizeof(struct curandStateMtgp32));

    return false;
  }
  this->ptr_array_cuRAND_State_MTGP32_shuffle =
      tmp_ptr_array_cuRAND_State_MTGP32_shuffle;
  // |END| allocate cuRAND State MTGP32 shuffle. |END|

  // copy cuRAND State MTGP32 shuffle.
  Memory::Copy_Loop<struct curandStateMtgp32>(
      ptr_curandStateMtgp32_received,
      ptr_curandStateMtgp32_received + number_states_MTGP32_received,
      this->ptr_array_cuRAND_State_MTGP32_shuffle);
  // |END| copy cuRAND State MTGP32 shuffle. |END|

  // allocate tmp_ptr_array_mtgp32_kernel_params_t.
  tmp_ptr_array_mtgp32_kernel_params_t =
      new struct mtgp32_kernel_params[number_states_MTGP32_received];
  if (tmp_ptr_array_mtgp32_kernel_params_t == nullptr) {
    ERR(L"Can not allocate %zu bytes.",
                 static_cast<size_t>(number_states_MTGP32_received) *
                     sizeof(struct mtgp32_kernel_params));

    return false;
  }
  // |END| allocate tmp_ptr_array_mtgp32_kernel_params_t. |END|

  // Assign cuRAND State MTGP32 shuffle variable.
  struct dim3 tmp_dim3_grid(1, 1, 1u), tmp_dim3_block(1, 1, 1u);

  if (USE_PARALLEL && number_states_MTGP32_received >= warpSize) {
    this->Get__Class_Device_Information_Array()
        ->Get__CUDA_Device()
        ->Grid_Block_1Dimensions(
            static_cast<size_t>(number_states_MTGP32_received), 0_UZ,
            tmp_dim3_grid, tmp_dim3_block);
  }

  cuRAND__Memcpy_cuRAND_State_MTGP32(
      number_states_MTGP32_received, tmp_ptr_array_cuRAND_State_MTGP32_shuffle,
      ptr_curandStateMtgp32_received, tmp_ptr_array_mtgp32_kernel_params_t,
      &tmp_dim3_grid, &tmp_dim3_block);

  this->p_number_cuRAND_State_MTGP32_shuffle = number_states_MTGP32_received;
  // |END| Assign cuRAND State MTGP32 shuffle variable. |END|

  return true;
}

template <typename T>
__global__ void kernel__CUDA_Dataset_Manager__Total_Blocks_cuRAND_MTGP32(
    int *const ptr_number_states_MTGP32_received,
    class cuDatasets<T> const *const ptr_CUDA_Dataset_Manager_received) {
  double const tmp_number_blocks(
      ceil(static_cast<double>(
               ptr_CUDA_Dataset_Manager_received->get_n_data()) /
           static_cast<double>(ptr_CUDA_Dataset_Manager_received
                                   ->Get__Class_Device_Information_Array()
                                   ->Get__CUDA_Device()
                                   ->Get__Warp_Size())));

  if (tmp_number_blocks > (std::numeric_limits<int>::max)()) {
    ERR(
        L"Overflow conversion (%f) to int (%d).", tmp_number_blocks, (std::numeric_limits<int>::max)());
  }

  *ptr_number_states_MTGP32_received =
      static_cast<int>(ceil(tmp_number_blocks / 256.0));
}

template <typename T>
__host__ bool cuDatasets<T>::Initialize_cuRAND(size_t const seed) {
  int tmp_number_states_MTGP32, *tmp_ptr_device_number_states_MTGP32(nullptr);

  CUDA__Safe_Call(
      cudaMalloc((void **)&tmp_ptr_device_number_states_MTGP32, sizeof(int)));

  kernel__CUDA_Dataset_Manager__Total_Blocks_cuRAND_MTGP32<T>
      <<<1, 1u>>>(tmp_ptr_device_number_states_MTGP32, this);

  CUDA__Check_Error();

  CUDA__Safe_Call(cudaMemcpy(&tmp_number_states_MTGP32,
                             tmp_ptr_device_number_states_MTGP32, sizeof(int),
                             cudaMemcpyKind::cudaMemcpyDeviceToHost));

  if (tmp_number_states_MTGP32 != 0) {
    struct mtgp32_kernel_params *tmp_ptr_mtgp32_kernel_params(NULL);

    struct curandStateMtgp32 *tmp_ptr_curandStateMtgp32_t(NULL);

    if (Allocate_cuRAND_MTGP32(tmp_number_states_MTGP32, seed,
                               tmp_ptr_mtgp32_kernel_params,
                               tmp_ptr_curandStateMtgp32_t) == false) {
      ERR(
          L"An error has been triggered from the "
          "\"Allocate_cuRAND_MTGP32(%d, %zu, ptr, ptr)\" function.", tmp_number_states_MTGP32, seed);

      CUDA__Safe_Call(cudaFree(tmp_ptr_device_number_states_MTGP32));

      return false;
    }

    kernel__CUDA_Dataset_Manager__Initialize_cuRAND_MTGP32<<<1, 1u>>>(
        tmp_number_states_MTGP32, tmp_ptr_curandStateMtgp32_t, this);

    CUDA__Check_Error();

    Cleanup_cuRAND_MTGP32(tmp_ptr_mtgp32_kernel_params,
                          tmp_ptr_curandStateMtgp32_t);
  }

  CUDA__Safe_Call(cudaFree(tmp_ptr_device_number_states_MTGP32));

  return true;
}

// template initialization declaration.
template class cuDatasets<var>;
