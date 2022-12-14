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

#include "deep-learning/data/enum/env.hpp"
#include "deep-learning/v1/data/enum/dataset.hpp"
#include "deep-learning/data/dataset.cuh"

#include <device_launch_parameters.h>

template <typename T>
class Datasets;

template <typename T>
class cuDatasets {
 protected:
  size_t p_n_data = 0;
  size_t p_seq_w = 0;
  size_t p_n_inp = 0;
  size_t p_n_out = 0;

  T **Xm = nullptr;
  T **Ym = nullptr;

  // cuRAND.
  int p_number_cuRAND_State_MTGP32_shuffle = 0;

  struct curandStateMtgp32 *ptr_array_cuRAND_State_MTGP32_shuffle = nullptr;
  // |END| cuRAND. |END|

  class cuDevicesProp *p_ptr_Class_Device_Information_Array = nullptr;  // Ptr.

 public:
  __host__ __device__ cuDatasets(void);
  __host__ __device__ ~cuDatasets(void);

  __host__ static void static_Deallocate_CUDA_Dataset_Manager(
      class cuDatasets<var> *&ptr_CUDA_Dataset_Manager_received);

  __host__ bool copy(class Datasets *const datasets);
  __device__ bool device_Copy(
      size_t const number_examples_received,
      size_t const number_inputs_received, size_t const number_outputs_received,
      size_t const number_recurrent_depth_received,
      T const *ptr_array_inputs_received, T const *ptr_array_outputs_received,
      class cuDeviceProp *const ptr_Class_Device_Information_received);
  __host__ bool Initialize_CUDA_Device(void);
  __host__ bool Initialize_cuRAND(size_t const seed);
  __device__ bool Initialize_cuRAND_MTGP32(
      int const size_received,
      struct curandStateMtgp32 *const ptr_curandStateMtgp32);
  __device__ bool Add_CUDA_Device(
      int const index_device_received,
      struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received);
  __host__ __device__ bool Deallocate(void);
  __host__ __device__ bool Initialize(void);
  __host__ __device__ bool Initialize(
      DL::ENV::TYPE const type_data_received,
      DL::DATASET::TYPE const
          type_gradient_descent_received);
  __host__ __device__ bool Initialize_Mini_Batch_Stochastic_Gradient_Descent(
      bool const use_shuffle_received,
      size_t const desired_number_examples_per_mini_batch_received,
      size_t const number_mini_batch_maximum_received);
  __host__ __device__ bool Initialize__Cross_Validation(
      bool const use_shuffle_received, size_t const number_k_fold_received,
      size_t const number_k_sub_fold_received);
  __host__ __device__ bool Initialize__Cross_Validation(void);
  __host__ __device__ bool Set__Type_Gradient_Descent(
      DL::ENV::TYPE const type_data_received,
      DL::DATASET::TYPE const
          type_gradient_descent_received);
  __host__ __device__ bool Prepare_Storage(void);
  __host__ __device__ bool Prepare_Storage(
      size_t const number_examples_training_received,
      size_t const number_examples_testing_received);
  __host__ __device__ bool Prepare_Storage(
      size_t const number_examples_training_received,
      size_t const number_examples_validation_received,
      size_t const number_examples_testing_received);
  __host__ __device__ bool Prepare_Storage(
      float const number_examples_percent_training_received,
      float const number_examples_percent_testing_received);
  __host__ __device__ bool Prepare_Storage(
      float const number_examples_percent_training_received,
      float const number_examples_percent_validation_received,
      float const number_examples_percent_testing_received);

  __host__ __device__ size_t get_n_data(void) const;
  __host__ __device__ size_t get_n_inp(void) const;
  __host__ __device__ size_t get_n_out(void) const;
  __host__ __device__ size_t get_seq_w(void) const;

  __device__ void train(float &ref_loss_received,
                           float &ref_accuracy_received,
                           class cuModel *const ptr_cuModel_received);
  __host__ float train(class Model *const model);
  __host__ float Type_Testing(
      DL::ENV::TYPE const type_data_received,
      class Model *const model);
  __device__ void device__Type_Testing(
      float &ref_loss_received, float &ref_accuracy_received,
      DL::ENV::TYPE const type_data_received,
      class cuModel *const ptr_cuModel_received);

  __host__ __device__ enum ENUM_TYPE_DATASET_MANAGER_STORAGE
  get_storage_type(void) const;

  __device__ T get_inp(size_t const index_received,
                             size_t const sub_index_received) const;
  __device__ T get_out(size_t const index_received,
                              size_t const sub_index_received) const;
  __device__ T *get_inp(size_t const index_received) const;
  __device__ T *get_out(size_t const index_received) const;
  __device__ T **Get__Input_Array(void) const;
  __device__ T **Get__Output_Array(void) const;

  __host__ __device__ size_t Get__Sizeof(void) const;

  __device__ class cuDataset<T> *get_dataset(
      DL::ENV::TYPE const type_storage_received) const;

  __device__ class cuDevicesProp *Get__Class_Device_Information_Array(
      void) const;

 private:
  enum ENUM_TYPE_DATASET_MANAGER_STORAGE _type_storage_data =
      ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_NONE;

  class cuDataset<T> *_ptr_array_Dataset = nullptr;
};