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

#include "deep-learning-lib/v1/data/enum/dataset.hpp"

#include <device_launch_parameters.h>

template <typename T>
class cuDatasets;

template <typename T>
class cuDataset {
 protected:
  size_t p_n_data = 0;
  size_t p_seq_w = 0;
  size_t p_n_inp = 0;
  size_t p_n_out = 0;

  T **Xm = nullptr;   // Size[D], Size[I].
  T **Ym = nullptr;  // Size[D], Size[O].

  struct dim3 *ptr_array_dim3_grid_batch = nullptr;              // Size[1].
  struct dim3 *ptr_array_dim3_block_batch = nullptr;             // Size[1].
  struct dim3 *ptr_array_dim3_grid_batch_fold = nullptr;         // Size[1].
  struct dim3 *ptr_array_dim3_block_batch_fold = nullptr;        // Size[1].
  struct dim3 *ptr_array_dim3_grid_shuffle = nullptr;            // Size[1].
  struct dim3 *ptr_array_dim3_block_shuffle = nullptr;           // Size[1].
  struct dim3 *ptr_array_dim3_grid_index_transposed = nullptr;   // Size[1].
  struct dim3 *ptr_array_dim3_block_index_transposed = nullptr;  // Size[1].

  DL::DATASET::TYPE p_type_dataset_process =
      DL::DATASET::NONE;

  class cuDevicesProp *p_ptr_Class_Device_Information_Array = nullptr;  // Ptr.

  // Mini-Batch Stochastic
  bool use_shuffle = true;

  size_t p_number_mini_batch = 0;
  size_t p_number_examples_mini_batch = 0;
  size_t p_number_examples_per_iteration = 0;
  size_t p_number_examples_last_iteration = 0;
  size_t *ptr_array_stochastic_index = nullptr;

  T **ptr_array_inputs_array_stochastic = nullptr;
  T **ptr_array_outputs_array_stochastic = nullptr;
  // - Mini-Batch Stochastic -

  // Cross Validation k-fold
  size_t number_examples_k_fold = 0;
  size_t number_k_fold = 0;
  size_t number_k_sub_fold = 0;
  size_t number_examples_per_fold = 0;
  size_t number_examples_training = 0;
  size_t number_examples_validating = 0;
  size_t number_examples_per_sub_iteration = 0;
  size_t number_examples_last_sub_iteration = 0;

  T **ptr_array_inputs_array_k_fold = nullptr;
  T **ptr_array_outputs_array_k_fold = nullptr;
  T **ptr_array_inputs_array_k_sub_fold = nullptr;
  T **ptr_array_outputs_array_k_sub_fold = nullptr;

  class cuDataset<T> *ptr_Validation_Dataset = nullptr;
  // - Cross Validation k-fold -

  // cuRAND.
  size_t p_number_cuRAND_State_MTGP32_shuffle = 0;
  size_t p_number_blocks_shuffle = 0;

  struct curandStateMtgp32 *ptr_array_cuRAND_State_MTGP32_shuffle = nullptr;
  // |END| cuRAND. |END|

 public:
  __host__ __device__ cuDataset(void);
  __host__ cuDataset(DL::DATASET_FORMAT::TYPE const
                         dset_fmt,
                     std::string const &ref_path_received);
  __host__ __device__ ~cuDataset(void);

  __device__ class cuDataset<T> &operator=(
      class cuDataset<T> const &ref_Dataset_received);

  __device__ void copy(class cuDataset<T> const &ref_Dataset_received);
  __device__ void copy(class cuDatasets<T> const &ref_Dataset_Manager_received);
  __device__ void reference(
      size_t const number_examples_received,
      size_t const number_inputs_received, size_t const number_outputs_received,
      size_t const number_recurrent_depth_received,
      T **const ptr_array_inputs_array_received,
      T **const ptr_array_outputs_array_received,
      size_t const number_cuRAND_State_MTGP32_shuffle_received,
      struct curandStateMtgp32 *const ptr_cuRAND_State_MTGP32_received,
      class cuDevicesProp *const ptr_Class_Device_Information_Array_received);
  __device__ void Train_Epoch_Batch(
      class cuModel *const ptr_cuModel_received);
  __device__ void Train_Batch_Batch(
      class cuModel *const ptr_cuModel_received);

  // Mini-Batch Stochastic
  __device__ void Train_Epoch_Mini_Batch_Stochastic(
      class cuModel *const ptr_cuModel_received);
  __device__ void Train_Batch_Mini_Batch_Stochastic(
      class cuModel *const ptr_cuModel_received);
  // - Mini-Batch Stochastic -

  // Cross Validation k-fold
  __device__ void Train_Epoch_Cross_Validation_K_Fold(
      class cuModel *const ptr_cuModel_received);
  __device__ void Train_Batch_Cross_Validation_K_Fold(
      class cuModel *const ptr_cuModel_received);
  // - Cross Validation k-fold -

  __device__ bool device_Allocate(
      size_t const number_examples_received,
      size_t const number_inputs_received, size_t const number_outputs_received,
      size_t const number_recurrent_depth_received,
      T const *ptr_array_inputs_received, T const *ptr_array_outputs_received,
      class cuDeviceProp *const ptr_Class_Device_Information_received);
  __device__ bool Allocate_Dim3(void);
  __host__ bool Initialize_CUDA_Device(void);
  __host__ bool Initialize_cuRAND(size_t const seed);
  __device__ bool Initialize_cuRAND_MTGP32(
      int const size_received,
      struct curandStateMtgp32 *const ptr_curandStateMtgp32);
  __device__ bool Add_CUDA_Device(
      int const index_device_received,
      struct cudaDeviceProp *const ptr_struct_cudaDeviceProp_received);
  __device__ bool valide_spec(
      size_t const &ref_number_inputs_received,
      size_t const &ref_number_outputs_received) const;
  __host__ __device__ bool Initialize(
      DL::DATASET::TYPE const
          type_gradient_descent_received);
  __host__ __device__ bool Initialize_Mini_Batch_Stochastic_Gradient_Descent(
      bool const use_shuffle_received,
      size_t const desired_number_examples_per_mini_batch_received,
      size_t const number_mini_batch_maximum_received);
  __host__ __device__ bool Initialize__Cross_Validation(
      bool const use_shuffle_received, size_t const number_k_fold_received,
      size_t const number_k_sub_fold_received,
      class cuDatasets<T> const *const ptr_CUDA_Dataset_Manager_received);
  __host__ __device__ bool Initialize__Cross_Validation(
      class cuDatasets<T> const *const ptr_CUDA_Dataset_Manager_received);
  __host__ __device__ bool Set__Type_Gradient_Descent(
      DL::DATASET::TYPE const
          type_gradient_descent_received);
  __host__ __device__ bool Deallocate(void);

  __host__ __device__ bool Get__Use__Shuffle(void) const;

  __host__ __device__ size_t Get__Total_Data(void) const;
  __host__ __device__ size_t get_n_data(void) const;
  __host__ __device__ size_t Get__Number_CV_K_Fold(void) const;
  __host__ __device__ size_t Get__Number_CV_K_Sub_Fold(void) const;
  __host__ __device__ size_t Get__Number_CV_Data_Per_Fold(void) const;
  __host__ __device__ size_t Get__Number_CV_Data_Training(void) const;
  __host__ __device__ size_t Get__Number_CV_Data_Validating(void) const;
  __host__ __device__ size_t Get__Number_CV_Data_Per_Sub_Iteration(void) const;
  __host__ __device__ size_t Get__Number_CV_Data_Last_Sub_Iteration(void) const;
  __host__ __device__ size_t get_n_inp(void) const;
  __host__ __device__ size_t get_n_out(void) const;
  __host__ __device__ size_t get_seq_w(void) const;

  __host__ float Training_Process_Batch(
      class Model *const model);
  __device__ void device__Training_Process_Batch(
      float &ref_loss_received, float &ref_accuracy_received,
      class cuModel *const ptr_cuModel_received);

  // Mini-Batch Stochastic
  __host__ float Training_Process_Mini_Batch_Stochastic(
      class Model *const model);
  __device__ void device__Training_Process_Mini_Batch_Stochastic(
      float &ref_loss_received, float &ref_accuracy_received,
      class cuModel *const ptr_cuModel_received);
  // - Mini-Batch Stochastic -

  // Cross Validation k-fold
  __host__ float Training_Process_Cross_Validation_K_Fold(
      class Model *const model);
  __device__ void device__Training_Process_Cross_Validation_K_Fold(
      float &ref_loss_received, float &ref_accuracy_received,
      class cuModel *const ptr_cuModel_received);
  // - Cross Validation k-fold -
  __host__ float evaluate(class Model *const model);
  __device__ void device__Testing(
      float &ref_loss_received, float &ref_accuracy_received,
      class cuModel *const ptr_cuModel_received);

  __host__ __device__ DL::DATASET::TYPE
  Get__Type_Dataset_Process(void) const;

  __device__ T get_inp(size_t const index_received,
                             size_t const sub_index_received) const;
  __device__ T get_out(size_t const index_received,
                              size_t const sub_index_received) const;
  __device__ T *get_inp(size_t const index_received) const;
  __device__ T *get_out(size_t const index_received) const;
  __device__ T **Get__Input_Array(void) const;
  __device__ T **Get__Output_Array(void) const;

  __host__ __device__ size_t Get__Sizeof(void) const;

  // Mini-Batch Stochastic
  __device__ void Mini_Batch_Stochastic__Reset(void);
  // - Mini-Batch Stochastic -

  // Cross Validation k-fold
  __device__ void Cross_Validation_K_Fold__Reset(void);
  // - Cross Validation k-fold -

  __device__ class cuDevicesProp *Get__Class_Device_Information_Array(
      void) const;

 private:
  // Mini-Batch Stochastic
  __device__ void Mini_Batch_Stochastic__Initialize_Shuffle(void);
  __device__ void Mini_Batch_Stochastic__Shuffle(void);
  __device__ bool Mini_Batch_Stochastic__Increment_Mini_Batch(
      size_t const mini_batch_iteration_received);
  // - Mini-Batch Stochastic -

  // Cross Validation k-fold
  __device__ void Cross_Validation_K_Fold__Initialize_Shuffle(void);
  __device__ void Cross_Validation_K_Fold__Shuffle(void);
  __device__ bool Cross_Validation_K_Fold__Increment_Fold(
      size_t const fold_received);
  __device__ bool Cross_Validation_K_Fold__Increment_Sub_Fold(
      size_t const fold_sub_received);
  __device__ float Test_Epoch_Cross_Validation_K_Fold(
      class cuModel *ptr_cuModel_received);
  // - Cross Validation k-fold -

  bool _reference = false;
};
