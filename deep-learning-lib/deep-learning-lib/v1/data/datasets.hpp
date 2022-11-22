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

#include "deep-learning-lib/v1/learner/model.hpp"
#include "deep-learning-lib/v1/data/scaler.hpp"
#include "deep-learning-lib/data/time.hpp"
#include "deep-learning-lib/data/enum/env.hpp"
#include "deep-learning-lib/v1/data/enum/dataset.hpp"
#include "deep-learning-lib/ops/distributions/bernoulli.hpp"
#include "deep-learning-lib/ops/distributions/real.hpp"
#include "deep-learning-lib/ops/distributions/gaussian.hpp"
#include "deep-learning-lib/ops/distributions/integer.hpp"
#include "deep-learning-lib/v1/ops/while.hpp"
#include "deep-learning-lib/ops/math.hpp"

namespace DL::v1 {
bool scan_datasets(DATASET_FORMAT::TYPE &ref_type_dateset_file_received,
                   std::wstring const &path_name);

class DatasetV1 {
 protected:
  size_t p_str_i = 0_UZ;

  size_t p_n_data = 0_UZ;

  size_t p_n_data_alloc = 0_UZ;

  size_t p_seq_w = 0_UZ;

  size_t p_n_inp = 0_UZ;

  size_t p_n_out = 0_UZ;

  size_t p_file_buffer_size = 32_UZ * MEGABYTE;  // byte(s).

  size_t p_file_buffer_shift_size = 256_UZ * KILOBYTE;  // byte(s).

  size_t *Xm_coeff_size = nullptr;
  size_t *Ym_coeff_size = nullptr;

  real const **Xm = nullptr;
  real const **Ym = nullptr;

  real *X = nullptr;
  real *Y = nullptr;

  real **Xm_coeff = nullptr;
  real **Ym_coeff = nullptr;

  DATASET::TYPE p_type_dataset_process = DATASET::BATCH;

  DATASET_FORMAT::TYPE p_type_data_file = DATASET_FORMAT::SPLIT;

  ENV::TYPE env_type = ENV::NONE;

 public:
  DatasetV1(void);

  DatasetV1(DATASET_FORMAT::TYPE const dset_fmt,
          std::wstring const &path_name);
  
  DatasetV1(DATASET_FORMAT::TYPE const dset_fmt,
          ENV::TYPE const env_type,
          std::wstring const &path_name);

  virtual ~DatasetV1(void);

  DatasetV1 &operator=(DatasetV1 const &cls);

  void copy(DatasetV1 const &cls);

  void reference(size_t const number_examples_received,
                 real const **Xm,
                 real const **Ym,
                 DatasetV1 const &cls);

  void Train_Epoch_OpenMP(Model *const model);

  void Train_Epoch_Loop(Model *const model);

  void Check_Use__Label(void);

  void Compute__Start_Index(void);

  void Adept__Gradient(Model *const model);

  virtual void Train_Batch_BP_OpenMP(Model *const model);

  virtual void Train_Batch_BP_Loop(Model *const model);

  virtual bool Initialize(void);

  bool Set__Type_Data_File(
      DATASET_FORMAT::TYPE const type_dataset_file_received);

  bool save(std::wstring const &path_name,
            bool const normalize_received = false);

  /* Save the output of an autoencoder to a file. */
  bool save(Model *const model, std::wstring const path_received);

  bool shift(size_t const shift, DATA::TYPE const data_type);

  bool Time_Direction(real const minimum_range_received,
                      real const maximum_range_received,
                      DATA::TYPE const data_type);

  bool Input_To_Output(DATA::TYPE const data_type);

  bool Unrecurrent(void);

  bool Remove(size_t const input_index_received, DATA::TYPE const data_type);

  bool allocate(DATASET_FORMAT::TYPE const dset_fmt,
                std::wstring const &path_name);

  bool Remove_Duplicate(void);

  bool Spliting_Dataset(size_t const desired_data_per_file_received,
                        std::wstring const &ref_path_file_received);

  bool simulate_trading(Model *const model);

  bool replace_entries(DatasetV1 const *const dataset,
                       DATA::TYPE const data_type);

  /* Replace input by the output of an autoencoder. */
  bool replace_entries(DatasetV1 const *const dataset,
                       Model *const model);

  bool Concat(DatasetV1 const *const ptr_source_Dataset_received);

  bool Save__Dataset_Custom(std::wstring const &path_name);

  bool Save__Sequential_Input(size_t const number_recurrent_depth_received,
                              std::wstring const &path_name);

  bool preprocess_minmax(size_t const data_start_index_received,
                         size_t const data_end_index_received,
                         real const minval,
                         real const maxval,
                         real const minimum_range_received,
                         real const maximum_range_received,
                         DATA::TYPE const data_type);

  bool preprocess_minmax(size_t const data_start_index_received,
                         size_t const data_end_index_received,
                         size_t const input_index_received,
                         real const minval,
                         real const maxval,
                         real const minimum_range_received,
                         real const maximum_range_received,
                         DATA::TYPE const data_type);

  bool preprocess_minmax(real *const ptr_array_inputs_received,
                         DATA::TYPE const data_type);

  bool preprocess_minmax(size_t const input_index_received,
                         real *const ptr_array_inputs_received,
                         DATA::TYPE const data_type);

  bool preprocess_minmax_inv(DATA::TYPE const data_type);

  bool preprocess_minmax_inv(size_t const input_index_received,
                             DATA::TYPE const data_type);

  bool preprocess_minmax_inv(real *const ptr_array_inputs_received,
                             DATA::TYPE const data_type);

  bool preprocess_minmax_inv(size_t const input_index_received,
                             real *const ptr_array_inputs_received,
                             DATA::TYPE const data_type);

  bool Preprocessing__Zero_Centered(size_t const data_start_index_received,
                                    size_t const data_end_index_received,
                                    real const multiplier_received,
                                    DATA::TYPE const data_type);

  bool Preprocessing__Zero_Centered(size_t const data_start_index_received,
                                    size_t const data_end_index_received,
                                    size_t const input_index_received,
                                    real const multiplier_received,
                                    DATA::TYPE const data_type);

  bool Preprocessing__Zero_Centered(real *const ptr_array_inputs_received,
                                    DATA::TYPE const data_type);

  bool Preprocessing__Zero_Centered(size_t const input_index_received,
                                    real *const ptr_array_inputs_received,
                                    DATA::TYPE const data_type);

  bool Preprocessing__Zero_Centered_Inverse(DATA::TYPE const data_type);

  bool Preprocessing__Zero_Centered_Inverse(size_t const input_index_received,
                                            DATA::TYPE const data_type);

  bool preprocess_modwt(size_t const desired_J_level_received,
                        DATA::TYPE const data_type);

  bool preprocess_modwt(size_t const input_index_received,
                        size_t const desired_J_level_received,
                        DATA::TYPE const data_type);

  /* Preprocess the input(s) array with the past datapoint in the dataset.
     The array should not be present inside the dataset!
     Should be call sequentialy w.r.t dataset order. */
  bool preprocess_modwt(size_t const input_index_received,
                        real *const ptr_array_inputs_received,
                        DATA::TYPE const data_type);

  bool preprocess_modwt_inv(DATA::TYPE const data_type);

  bool preprocess_modwt_inv(size_t const input_index_received,
                            DATA::TYPE const data_type);

  bool Preprocessing__Merge__MODWT(size_t const desired_J_level_received,
                                   DATA::TYPE const data_type);

  bool Preprocessing__Merge__MODWT(size_t const input_index_received,
                                   size_t const desired_J_level_received,
                                   DATA::TYPE const data_type);

  bool Preprocessing__Merge__MODWT(size_t const input_index_received,
                                   size_t const input_size_received,
                                   real *&ptr_array_inputs_received,
                                   DATA::TYPE const data_type);

  bool Preprocessing__Sequence_Window(size_t const sequence_window_received,
                                      size_t const sequence_horizon_received,
                                      real *&ptr_array_inputs_received);

  bool valide_spec(size_t const number_inputs_received,
                   size_t const number_outputs_received,
                   size_t const number_recurrent_depth_received) const;

  // Check if the object is a reference.
  bool Get__Reference(void) const;

  bool Use__Multi_Label(void) const;

  virtual bool Deallocate(void);

  virtual size_t get_n_data(void) const;

  virtual size_t get_n_batch(void) const;

  size_t get_n_inp(void) const;

  size_t get_n_out(void) const;

  size_t get_seq_w(void) const;

  size_t MODWT__J_Level_Maximum(void) const;

  double compute_accuracy(
      size_t const batch_size,
      real const *const *const ptr_array_inputs_received,
      real const *const *const ptr_array_desired_outputs_received,
      Model *const model);

  virtual double train(Model *const model);
  virtual double train_mp(Model *const model);
  virtual double train_st(Model *const model);

  virtual double evaluate(Model *const model);
  virtual double evaluate_mp(Model *const model);
  virtual double evaluate_st(Model *const model);

  DATASET::TYPE Get__Type_Dataset_Process(void) const;

  real get_min(size_t const data_start_index_received,
               size_t const data_end_index_received,
               size_t const input_index_received,
               DATA::TYPE const data_type) const;

  real get_min(size_t const data_start_index_received,
               size_t const data_end_index_received,
               DATA::TYPE const data_type) const;

  real get_max(size_t const data_start_index_received,
               size_t const data_end_index_received,
               size_t const input_index_received,
               DATA::TYPE const data_type) const;

  real get_max(size_t const data_start_index_received,
               size_t const data_end_index_received,
               DATA::TYPE const data_type) const;

  virtual real get_inp(size_t const index_received,
                       size_t const sub_index_received) const;

  virtual real get_out(size_t const index_received,
                       size_t const sub_index_received) const;

  virtual real const *const get_inp(size_t const index_received) const;

  virtual real const *const get_out(size_t const index_received) const;

  virtual real const *const *const Get__Input_Array(void) const;

  virtual real const *const *const Get__Output_Array(void) const;

  size_t Get__Sizeof(void);

  ScalerMinMax *const get_scaler_minmax(DATA::TYPE const data_type) const;

  ScalerZeroCentered *const Get__Scalar__Zero_Centered(
      DATA::TYPE const data_type) const;

 private:
  bool shift_arrays(size_t const input_index_received,
                    size_t const shift_size_received,
                    DATA::TYPE const data_type);

  bool Save__Dataset(std::wstring const &path_name,
                     bool const normalize_received = false);

  bool save_split_XY(std::wstring const &path_name);

  bool save_X(std::wstring const &path_name);

  bool save_Y(std::wstring const &path_name);

  bool Allocate__Dataset(std::wstring const &path_name);

  bool Allocate__Dataset_Split(std::wstring const &path_name);

  bool Allocate__Dataset_Split__Input(std::wstring const &path_name);

  bool Allocate__Dataset_Split__Output(std::wstring const &path_name);

  bool Allocate__MNIST(std::wstring const &path_name);

  bool _reference = false;

  bool _use_multi_label = false;

  ScalerMinMax *_ptr_input_array_scaler__minimum_maximum = nullptr;

  ScalerMinMax *_ptr_output_array_scaler__minimum_maximum = nullptr;

  ScalerZeroCentered *_ptr_input_array_scaler__zero_centered = nullptr;

  ScalerZeroCentered *_ptr_output_array_scaler__zero_centered = nullptr;
};

class MiniBatch : public DatasetV1 {
 public:
  MiniBatch(void);
  MiniBatch(bool const use_shuffle_received,
            size_t const desired_number_examples_per_mini_batch_received,
            size_t const number_mini_batch_maximum_received,
            DatasetV1 &dataset);
  virtual ~MiniBatch(void);

  void Shuffle(void);
  void Set__Use__Shuffle(bool const use_shuffle_received);
  void reset(void);

  virtual bool Initialize(void);
  bool Initialize(bool const use_shuffle_received,
                  size_t const desired_number_examples_per_mini_batch_received,
                  size_t const number_mini_batch_maximum_received);
  bool Set__Desired_Data_Per_Batch(
      size_t const desired_number_examples_per_mini_batch_received,
      size_t const number_mini_batch_maximum_received = 0_UZ);
  bool Increment_Mini_Batch(size_t const mini_batch_iteration_received);
  bool Get__Use__Shuffle(void) const;
  virtual bool Deallocate(void);
  bool use_shuffle = true;

  virtual size_t get_n_data(void) const;
  virtual size_t get_n_batch(void) const;
  size_t Get__Number_Examples_Per_Batch(void) const;
  size_t Get__Number_Examples_Last_Batch(void) const;
  size_t number_examples = 0_UZ;
  size_t number_mini_batch = 0_UZ;
  size_t number_examples_per_iteration = 0_UZ;
  size_t number_examples_last_iteration = 0_UZ;
  size_t *ptr_array_stochastic_index = nullptr;

  virtual double train_mp(Model *const model);
  virtual double train_st(Model *const model);

  virtual real get_inp(size_t const index_received,
                       size_t const sub_index_received) const;
  virtual real get_out(size_t const index_received,
                       size_t const sub_index_received) const;
  virtual real const *const get_inp(size_t const index_received) const;
  virtual real const *const get_out(size_t const index_received) const;
  virtual real const *const *const Get__Input_Array(void) const;
  virtual real const *const *const Get__Output_Array(void) const;
  real const **ptr_array_inputs_array_stochastic = nullptr;
  real const **ptr_array_outputs_array_stochastic = nullptr;

  Dist::Integer<size_t> Generator_Random;
};

class CrossVal : public DatasetV1 {
 protected:
  double Test_Epoch_OpenMP(Model *const model);
  double Test_Epoch_Loop(Model *const model);

 public:
  CrossVal(void);
  virtual ~CrossVal(void);

  void Shuffle(void);
  void Set__Use__Shuffle(bool const use_shuffle_received);
  void reset(void);

  virtual bool Initialize(void);
  /* number_k_sub_fold:
      = 0: number_k_fold - 1.
      = 1: no mini batch fold.
      > 1: mini batch fold. */
  bool Initialize__Fold(bool const use_shuffle_received,
                        size_t const number_k_fold_received,
                        size_t const number_k_sub_fold_received);
  bool Set__Desired_K_Fold(size_t const number_k_fold_received,
                           size_t const number_k_sub_fold_received);
  bool Increment_Fold(size_t const fold_received);
  bool Increment_Sub_Fold(size_t const sub_fold_received);
  bool Get__Use__Shuffle(void) const;
  virtual bool Deallocate(void);
  bool use_shuffle = true;

  virtual size_t get_n_data(void) const;
  virtual size_t get_n_batch(void) const;
  size_t Get__Number_Sub_Batch(void) const;
  size_t Get__Number_Examples_Training(void) const;
  size_t Get__Number_Examples_Validating(void) const;
  size_t Get__Number_Examples_Per_Fold(void) const;
  size_t Get__Number_Examples_Per_Sub_Iteration(void) const;
  size_t Get__Number_Examples_Last_Sub_Iteration(void) const;
  size_t number_examples = 0_UZ;
  size_t number_k_fold = 0_UZ;
  size_t number_k_sub_fold = 0_UZ;
  size_t number_examples_per_fold = 0_UZ;
  size_t number_examples_training = 0_UZ;
  size_t number_examples_validating = 0_UZ;
  size_t number_examples_per_sub_iteration = 0_UZ;
  size_t number_examples_last_sub_iteration = 0_UZ;
  size_t *ptr_array_stochastic_index = nullptr;

  virtual double train_mp(Model *const model);
  virtual double train_st(Model *const model);

  virtual real get_inp(size_t const index_received,
                       size_t const sub_index_received) const;
  virtual real get_out(size_t const index_received,
                       size_t const sub_index_received) const;
  virtual real const *const get_inp(size_t const index_received) const;
  virtual real const *const get_out(size_t const index_received) const;
  virtual real const *const *const Get__Input_Array(void) const;
  virtual real const *const *const Get__Output_Array(void) const;
  real const **ptr_array_inputs_array_k_fold = nullptr;
  real const **ptr_array_outputs_array_k_fold = nullptr;
  real const **ptr_array_inputs_array_k_sub_fold = nullptr;
  real const **ptr_array_outputs_array_k_sub_fold = nullptr;
  real const **ptr_array_inputs_array_validation = nullptr;
  real const **ptr_array_outputs_array_validation = nullptr;

  Dist::Integer<size_t> Generator_Random;
};

class Gaussian_Search {
 protected:
  Datasets **p_ptr_array_ptr_dataset_manager = nullptr;

  Model **individuals = nullptr;

 public:
  Gaussian_Search(void);
  ~Gaussian_Search(void);

  void Deallocate__Dataset_Manager(void);
  void Deallocate__Population(void);
  void Deallocate(void);

  bool Initialize__OpenMP(void);
  bool set_mp(bool const use_openmp_received);
  bool Set__Population_Size(size_t const population_size_received);
  bool Set__Population_Gaussian(
      double const population_gaussian_percent_received);
  bool Set__Maximum_Thread_Usage(
      double const percentage_maximum_thread_usage_received);
  bool Allouable__Thread_Size(size_t const desired_number_threads_received,
                              size_t &ref_number_threads_allouable_received);
  bool update_mem_thread_size(size_t const desired_number_threads_received);
  bool Update__Thread_Size__Population(
      size_t const desired_number_threads_received);
  bool Update__Batch_Size__Population(size_t const desired_batch_size_received);
  bool Update__Population(Model *const ptr_source_Neural_Network_received);
  bool Update__Dataset_Manager(
      Datasets *const ptr_source_Dataset_Manager_received);
  bool Optimize(size_t const number_iterations_received,
                Datasets *const datasets, Model *const model);
  bool Evaluation(void);
  bool Evaluation(Datasets *const datasets);
  bool user_controls(void);
  bool User_Controls__Push_Back(void);
  bool User_Controls__Hyperparameter_Manager(void);
  bool User_Controls__OpenMP(void);
  // Index: Layer/Unit index.
  bool push_back(int const hyper_parameter_id_received,
                 size_t const index_received, real const value_received,
                 real const minval,
                 real const maxval,
                 real const variance = 0.1_r);
  bool Initialize__Hyper_Parameters(Model *const model);
  bool Initialize__Hyper_Parameter(
      std::tuple<int, size_t, real, real, real, real>
          &ref_hyperparameter_tuple_received,
      Model *const model);
  bool Shuffle__Hyper_Parameter(void);
  bool Feed__Hyper_Parameter(void);
  bool Deinitialize__OpenMP(void);

  bool Feed__Hyper_Parameter(
      std::tuple<int, size_t, real, real, real, real> const
          &ref_hyperparameter_tuple_received,
      Model *const model);

  std::wstring Get__ID_To_String(int const hyperparameter_id_received) const;

 private:
  bool Enable__OpenMP__Population(void);
  bool Disable__OpenMP__Population(void);

  bool Allocate__Population(size_t const population_size_received);
  bool Allocate__Thread(size_t const number_threads_received);
  bool Reallocate__Population(size_t const population_size_received);
  bool Reallocate__Thread(size_t const number_threads_received);
  bool Optimize__Loop(size_t const number_iterations_received,
                      Datasets *const datasets, Model *const model);
  bool Optimize__OpenMP(size_t const number_iterations_received,
                        Datasets *const datasets, Model *const model);
  bool Evaluation__Loop(Datasets *const datasets);
  bool Evaluation__OpenMP(Datasets *const datasets);
  bool _use_mp = false;
  bool _is_mp_initialized = false;

  size_t _population_size = 0_UZ;
  size_t _number_threads = 0_UZ;
  size_t _cache_number_threads = 0_UZ;

  double _population_gaussian_percent =
      60.0;  // Exploitation population. remaining exploration.
  double _percentage_maximum_thread_usage = 100.0;
  double _cache_maximum_threads_percent = 0.0;

  std::vector<std::tuple<int, size_t, real, real, real, real>>
      _vector_hyperparameters;

  /* std::tuple<
                      [0]: ID,
                      [1]: Layer/Unit index,
                      [2]: Value,
                      [3]: Value Minimum,
                      [4]: Value Maximum,
                      [5]: Variance
                                          > */
  std::tuple<int, size_t, real, real, real, real>
      *_ptr_selected_hyperparameter = nullptr;

  Datasets *p_ptr_array_dataset_manager = nullptr;

  Model *p_ptr_array_individuals = nullptr;

  Dist::Integer<int> int_gen;  // Index generator.
  Dist::Real real_gen;
  Dist::Gaussian gaussian;
};

class HyperOpt {
 protected:
  size_t p_number_hyper_optimization_iterations = 10_UZ;
  size_t p_number_hyper_optimization_iterations_delay = 25_UZ;
  size_t p_optimization_iterations_since_hyper_optimization = 0_UZ;

 public:
  HyperOpt(void);
  virtual ~HyperOpt(void);

  void reset(void);
  void Deallocate__Gaussian_Search(void);

  bool Set__Hyperparameter_Optimization(
      HYPEROPT::TYPE const type_hyper_optimization_received);
  bool Set__Number_Hyperparameter_Optimization_Iterations(
      size_t const number_hyper_optimization_iterations_delay_received);
  bool Set__Number_Hyperparameter_Optimization_Iterations_Delay(
      size_t const number_hyper_optimization_iterations_delay_received);
  bool Get__Evaluation_Require(void) const;
  bool Optimize(Datasets *const datasets, Model *const model);
  bool Evaluation(void);
  bool Evaluation(Datasets *const datasets);
  bool user_controls(void);
  bool User_Controls__Change__Hyperparameter_Optimization(void);
  bool allocate_gaussian_opt(void);
  bool Deallocate(void);

  double optimize(Datasets *const datasets, Model *const model);

  HYPEROPT::TYPE Get__Hyperparameter_Optimization(void) const;

 private:
  bool _evaluation_require = false;

  Gaussian_Search *gaussian_opt = nullptr;

  HYPEROPT::TYPE _type_hyperparameter_optimization = HYPEROPT::NONE;
};

class CrossValOpt : public CrossVal, public HyperOpt {
 public:
  CrossValOpt(void);
  virtual ~CrossValOpt(void);

  virtual double train_mp(Model *const model);
  virtual double train_st(Model *const model);

  bool Deallocate(void);
};

struct DatasetParams {
 public:
  /* value_0:
          [-1]: User choose,
          [Mini-batch stochastic gradient descent]: shuffle,
          [Cross-validation]: shuffle */
  int value_0 = -1;
  /* value_1:
          [-1]: User choose,
          [Mini-batch stochastic gradient descent]:
     number_desired_data_per_batch, [Cross-validation]: number_k_fold */
  int value_1 = -1;
  /* value_2:
          [-1]: User choose,
          [Mini-batch stochastic gradient descent]: number_maximum_batch,
          [Cross-validation]: number_k_sub_fold */
  int value_2 = -1;
  /* value_3:
          [-1]: User choose,
          [Cross-validation, hyper-optimization]: Number hyperparameter
     optimization iteration(s) */
  int value_3 = -1;
  /* value_3:
          [-1]: User choose,
          [Cross-validation, hyper-optimization]: Number hyperparameter
     optimization iteration(s) delay */
  int value_4 = -1;
};

struct DatasetsParams {
 public:
  /* Type storage:
          [-1]: User choose.
          [0]: Training.
          [1]: Training and testing.
          [2]: Training, validation and testing. */
  int type_storage = -1;
  /* Type training:
          [-1]: User choose.
          [0]: Batch gradient descent.
          [1]: Mini-batch.
          [2]: Cross-validation.
          [3]: Cross-validation, random search. */
  int type_train = -1;

  double pct_train_size = 0.0;
  double pct_valid_size = 0.0;

  DatasetParams train_params;
};

struct Data_Accuracy {
  Data_Accuracy(void) {}
  ~Data_Accuracy(void) {}

  Data_Accuracy &operator=(
      Data_Accuracy const &cls) {
    if (&cls != this) {
      this->desired_accuracy =
          cls.desired_accuracy;
      this->ptr_array_desired_entries =
          cls.ptr_array_desired_entries;
    }

    return *this;
  }

  double desired_accuracy = 100_r;
  double *ptr_array_desired_entries = nullptr;
};

class Datasets : public DatasetV1, public HyperOpt {
 public:
  Datasets(void);
  Datasets(DATASET_FORMAT::TYPE const dset_fmt,
           std::wstring const &path_name);
  virtual ~Datasets(void);

  std::vector<ENV::TYPE> envs_type_evalt;
  bool use_valid_set = false;
  bool use_testg_set = false;

  void evaluate_envs(Model *const model);
  void set_eval_env(ENV::TYPE const type_evaluation_received);
  void set_desired_optimization_time_between_reports(
      double const desired_optimization_time_between_reports_received);
  void optimize(
      WhileCond const &while_cond, bool const save_trainer,
      bool const save_trained, double const desired_loss,
      std::wstring const &ref_path_net_trainer_neural_network_received,
      std::wstring const &ref_path_nn_trainer_neural_network_received,
      std::wstring const &ref_path_net_trained_neural_network_received,
      std::wstring const &ref_path_nn_trained_neural_network_received,
      Model *&ptr_trainer_Neural_Network_received,
      Model *&ptr_trained_Neural_Network_received);
  void Optimization__Testing(bool const report, TIME_POINT &time_str,
                             TIME_POINT &time_end, Model *&model);
  void Deallocate__Storage(void);

  bool Set__Maximum_Data(size_t const number_examples_received);
  bool Reallocate_Internal_Storage(void);
  bool push_back(real const *const ptr_array_inputs_received,
                 real const *const ptr_array_outputs_received);
  bool Prepare_Storage(DatasetV1 *const ptr_TrainingSet_received);
  bool Prepare_Storage(size_t const number_examples_training_received,
                       size_t const number_examples_testing_received,
                       DatasetV1 *const ptr_TrainingSet_received,
                       DatasetV1 *const ptr_TestingSet_received);
  bool Prepare_Storage(size_t const number_examples_training_received,
                       size_t const number_examples_validation_received,
                       size_t const number_examples_testing_received,
                       DatasetV1 *const ptr_TrainingSet_received,
                       DatasetV1 *const ptr_ValidatingSet_received,
                       DatasetV1 *const ptr_TestingSet_received);
  bool Prepare_Storage(double const number_examples_percent_training_received,
                       double const number_examples_percent_testing_received,
                       DatasetV1 *const ptr_TrainingSet_received,
                       DatasetV1 *const ptr_TestingSet_received);
  bool Prepare_Storage(double const number_examples_percent_training_received,
                       double const number_examples_percent_validation_received,
                       double const number_examples_percent_testing_received,
                       DatasetV1 *const ptr_TrainingSet_received,
                       DatasetV1 *const ptr_ValidatingSet_received,
                       DatasetV1 *const ptr_TestingSet_received);
  bool Initialize_Dataset(
      ENV::TYPE const env_type,
      DATASET::TYPE const type_dataset_process_received,
      DatasetParams const *const ptr_Dataset_Parameters_received = nullptr);
  bool Preparing_Dataset_Manager(
      DatasetsParams const *const ptr_Dataset_Manager_Parameters_received =
          nullptr);
  bool reference(Datasets *const ptr_source_Dataset_Manager_received);
  bool Copy__Storage(Datasets const *const ptr_source_Dataset_Manager_received);
  bool user_controls(void);
  bool User_Controls__Set__Maximum_Data(void);
  bool User_Controls__Type_Evaluation(void);
  bool User_Controls__Type_Metric(void);
  bool User_Controls__Optimization_Processing_Parameters(void);
  bool User_Controls__Optimization_Processing_Parameters__Batch(void);
  bool User_Controls__Optimization_Processing_Parameters__Mini_Batch(void);
  bool User_Controls__Optimization_Processing_Parameters__Cross_Validation(
      void);
  bool
  User_Controls__Optimization_Processing_Parameters__Cross_Validation__Gaussian_Search(
      void);
  bool User_Controls__Optimization(Model *&ptr_trainer_Neural_Network_received,
                                   Model *&ptr_trained_Neural_Network_received);
  bool Get__Dataset_In_Equal_Less_Holdout_Accepted(void) const;
  bool Use__Metric_Loss(void) const;
  virtual bool Deallocate(void);

  virtual double train(Model *const model);
  double Optimize(Model *const model);
  double Type_Testing(ENV::TYPE const env_type,
                      Model *const model);
  std::pair<double, double> Type_Update_Batch_Normalization(
      ENV::TYPE const env_type, Model *const model);
  double evaluate(Model *const model);
  double Get__Minimum_Loss_Holdout_Accepted(void) const;

  enum ENUM_TYPE_DATASET_MANAGER_STORAGE get_storage_type(void) const;

  DatasetV1 *Allocate__Dataset(DATASET::TYPE const type_dataset_process_received,
                             ENV::TYPE const type_data_received);
  DatasetV1 *get_dataset(ENV::TYPE const env_type) const;

  void Deallocate_CUDA(void);

  bool Initialize__CUDA(void);
  bool Initialize_Dataset_CUDA(ENV::TYPE const env_type);

#ifdef COMPILE_CUDA
  cuDatasets *Get__CUDA(void);
#endif

  ENV::TYPE Get__Type_Dataset_Evaluation(void) const;

 private:
  bool _reference = false;
  bool _dataset_in_equal_less_holdout_accepted = true;
  bool _use_metric_loss = true;

  size_t _maximum_examples = 0_UZ;

  double _minimum_loss_holdout_accepted = HUGE_VAL;
  double _size_dataset_training__percent = 60.0;
  double _size_dataset_validation__percent = 20.0;
  double _size_dataset_testing__percent = 20.0;
  double _desired_optimization_time_between_reports = 1.0;  // Seconds

  ENV::TYPE _type_evaluation = ENV::VALID;
  enum ENUM_TYPE_DATASET_MANAGER_STORAGE _type_storage_data =
      ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_NONE;

  DatasetV1 **_ptr_array_ptr_Dataset = nullptr;

#ifdef COMPILE_CUDA
  cuDatasets *_ptr_CUDA_Dataset_Manager = NULL;
#endif
};
}  // namespace DL