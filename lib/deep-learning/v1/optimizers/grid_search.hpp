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

#include "deep-learning/v1/learner/models.hpp"

namespace DL::v1 {
struct Dropout_Initializer__LW {
  bool operator==(Dropout_Initializer__LW const &cls) const;

  size_t layer_index = 0u;

  real value[2] = {0};

  LAYER_DROPOUT::TYPE type_layer_dropout = LAYER_DROPOUT::NONE;
};

struct Dropout_Initializer__Arguments {
  size_t layer_index = 0_UZ;

  real minval[2] = {0};
  real maxval[2] = {0};
  real step_size[2] = {0};

  LAYER_DROPOUT::TYPE type_layer_dropout = LAYER_DROPOUT::NONE;
};

class Grid_Search {
 protected:
  template <typename T>
  void push_back(std::vector<T> &vec, T &obj,
                 bool const duplicate_allowed = true);

  std::vector<real> p_vector_Weight_Decay;
  std::vector<real> p_vector_Max_Norm_Constraints;
  std::vector<real> p_vector_Normalization_Momentum_Average;

  std::vector<std::vector<Dropout_Initializer__LW>> p_vector_layers_Dropout;

 public:
  Grid_Search(void);
  ~Grid_Search(void);

  void Shuffle(void);

  bool link(Models &models);
  bool Set__Maximum_Iterations(size_t const maximum_iterations_received);
  bool Set__Use__Shuffle(bool const use_shuffle_received);
  bool push_back_wd(real const minval,
                               real const maxval,
                               real const step_size_received,
                               bool const allow_duplicate_received = true);
  bool push_back_max_norm_constraints(
      real const minval, real const maxval,
      real const step_size_received,
      bool const allow_duplicate_received = true);
  bool push_back_normalization_mom_avg(
      real const minval, real const maxval,
      real const step_size_received,
      bool const allow_duplicate_received = true);
  bool push_back_dropout(
      Dropout_Initializer__Arguments Dropout_Initializer__Arguments_received,
      bool const allow_duplicate_received = true);
  bool update_tree(void);
  bool user_controls(void);
  bool User_Controls__Shuffle(void);
  bool optimize(Models &ref_Neural_Network_Manager_received);
  bool Feed_Hyper_Parameters(size_t const hyper_parameters_index_received,
                             Model *const model);

 private:
  void Deallocate__Stochastic_Index(void);

  bool Allocate__Stochastic_Index(void);
  bool Reallocate__Stochastic_Index(size_t const total_iterations_received);
  bool _use_shuffle = false;

  size_t _maximum_iterations = 1u;

  size_t *_ptr_array_stochastic_index = nullptr;
  size_t _total_iterations = 0_UZ;

  std::vector<size_t> _vector_Tree;

  Dist::Integer<size_t> _Generator_Random_Integer;
};
}  // namespace DL