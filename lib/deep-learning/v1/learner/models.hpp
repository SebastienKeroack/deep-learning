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

#include "deep-learning/v1/data/scaler.hpp"
#include "deep-learning/v1/data/enum/hierarchy.hpp"
#include "deep-learning/v1/data/datasets.hpp"
#include "deep-learning/v1/learner/model.hpp"
#include "deep-learning/v1/ops/while.hpp"

#include <string>
#include <chrono>
#include <atomic>

namespace DL::v1 {
class Models {
 public:
  Models(void);
  ~Models(void);

  void Set__Auto_Save_Dataset(bool const auto_save_received);
  void auto_save_trainer(bool const auto_save_received);
  void auto_save_competitor(bool const auto_save_received);
  void auto_save_trained(bool const auto_save_received);
  void Set__Comparison_Expiration(size_t const expiration_seconds_received);
  void Deallocate__Neural_Network(HIERARCHY::TYPE const hierarchy);
  void Deallocate__Dataset_Manager(void);

  bool Set__Output_Mode(bool const use_last_layer_as_output_received,
                        HIERARCHY::TYPE const hierarchy);
  bool set_while_cond(WhileCond &while_cond);
  bool Set__Number_Inputs(size_t const number_inputs_received);
  bool Set__Number_Outputs(size_t const number_outputs_received);
  bool Set__Number_Recurrent_Depth(
      size_t const number_recurrent_depth_received);
  bool set_desired_loss(double const desired_loss_received);

  // [     GET      ]
  bool require_evaluate_envs(void) const;
  bool Get__Is_Output_Symmetric(void) const;
  bool Get__Path_Neural_Network_Exist(HIERARCHY::TYPE const hierarchy) const;

  size_t get_n_inp(HIERARCHY::TYPE const hierarchy) const;
  size_t get_n_out(HIERARCHY::TYPE const hierarchy) const;
  size_t get_seq_w(HIERARCHY::TYPE const hierarchy) const;

  double get_loss(HIERARCHY::TYPE const hierarchy,
               ENV::TYPE const type) const;
  double get_accu(HIERARCHY::TYPE const hierarchy, ENV::TYPE const type) const;

  var const *const Get__Output(size_t const time_step_index_received,
                               HIERARCHY::TYPE const hierarchy) const;

  std::wstring get_model_path(
      HIERARCHY::TYPE const hierarchy,
      std::wstring const path_postfix_received = L"net") const;
  std::wstring Get__Path_Dataset_Manager(void) const;

  Datasets *get_datasets(void);

  Model *get_model(HIERARCHY::TYPE const hierarchy);
  // ----- GET -----

  bool initialize_dirs(std::wstring const &ref_class_name_received);
  bool initialize_datasets(DatasetsParams const *const config = nullptr);
  bool create(size_t const maximum_allowable_host_memory_bytes_received);
  bool append_to_dataset(real const *const ptr_array_input_received,
                         real const *const ptr_array_output_received);
  bool Check_Expiration(void);
  bool evaluate_envs(void);
  bool evaluate_envs(Model *const ptr_neural_network_received);
  bool evaluate_envs_pre_train(void);
  bool evaluate_envs_pre_train(Model *const ptr_neural_network_received);
  bool pre_training(void);
  bool pre_training(std::vector<size_t> const
                        &ref_vector_epochs_per_pre_training_level_received);
  bool save_model(HIERARCHY::TYPE const hierarchy);
  bool if_require_evaluate_envs(void);
  bool if_require_evaluate_envs_pre_train(void);
  bool compare_trained(void);
  bool compare_trained_pre_train(void);

  void set_use_cuda(bool const use_cu);

  bool load_model(HIERARCHY::TYPE const hierarchy,
                  size_t const maximum_allowable_host_memory_bytes_received,
                  size_t const maximum_allowable_device_memory_bytes_received,
                  bool const copy_to_competitor_received);

  double optimize(void);

 private:
  bool _require_evaluate_envs = true;
  bool _auto_save_dataset = false;
  bool _optimization_auto_save_trainer = false;
  bool _optimization_auto_save_competitor = false;
  bool _optimization_auto_save_trained = false;

  size_t _n_inp = 0_UZ;
  size_t _n_out = 0_UZ;
  size_t _seq_w = 1_UZ;
  size_t _expiration_seconds = 24_UZ * 60_UZ * 60_UZ;

  double _desired_loss = 0.0_r;

  std::wstring _path_root = L"";
  std::wstring _path_trained_dir = L"";
  std::wstring _path_trainer_dir = L"";
  std::wstring _path_dataset_dir = L"";

  WhileCond _while_cond_opt;

  Model *_ptr_trainer_Neural_Network = nullptr;
  Model *_ptr_competitor_Neural_Network = nullptr;
  Model *_ptr_trained_Neural_Network = nullptr;

  Datasets *_ptr_Dataset_Manager = nullptr;

  std::chrono::system_clock::time_point _competitor_expiration =
      std::chrono::system_clock::now() +
      std::chrono::seconds(this->_expiration_seconds);

  bool _use_cuda = false;
};
}  // namespace DL