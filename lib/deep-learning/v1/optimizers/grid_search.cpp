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

#include "deep-learning/v1/data/datasets.hpp"
#include "deep-learning/data/string.hpp"
#include "deep-learning/data/time.hpp"
#include "deep-learning/ops/math.hpp"
#include "deep-learning/io/term/keyboard.hpp"
#include "deep-learning/io/logger.hpp"
#include "deep-learning/io/term/input.hpp"
#include "deep-learning/device/system/shutdown_block.hpp"
#include "deep-learning/v1/optimizers/grid_search.hpp"
#include "deep-learning/v1/mem/reallocate.hpp"

#include <iostream>
#include <array>

using namespace DL::Str;
using namespace DL::Term;

namespace DL::v1 {
bool Dropout_Initializer__LW::operator==(
    Dropout_Initializer__LW const &cls) const {
  if (this == &cls) return true;

  return (this->layer_index == cls.layer_index &&
          this->type_layer_dropout == cls.type_layer_dropout &&
          this->value[0] == cls.value[0] && this->value[1] == cls.value[1]);
}

Grid_Search::Grid_Search(void) {}

Grid_Search::~Grid_Search(void) { this->Deallocate__Stochastic_Index(); }

void Grid_Search::Shuffle(void) {
  size_t tmp_swap, i;
  size_t tmp_randomize_index;

  for (i = this->_total_iterations; i--;) {
    this->_Generator_Random_Integer.range(0, i);

    tmp_randomize_index = this->_Generator_Random_Integer();

    // Store the index to swap from the remaining index at "tmp_randomize_index"
    tmp_swap = this->_ptr_array_stochastic_index[tmp_randomize_index];

    // Get remaining index starting at index "i"
    // And store it to the remaining index at "tmp_randomize_index"
    this->_ptr_array_stochastic_index[tmp_randomize_index] =
        this->_ptr_array_stochastic_index[i];

    // Store the swapped index at the index "i"
    this->_ptr_array_stochastic_index[i] = tmp_swap;
  }
}

bool Grid_Search::link(Models &ref_Neural_Network_Manager) {
  size_t tmp_layer_index;

  real tmp_minimum, tmp_maximum, tmp_step_size;

  class Model *const tmp_ptr_Neural_Network(
      ref_Neural_Network_Manager.get_model(HIERARCHY::COMPETITOR));

  INFO(L"");

  if (this->Set__Maximum_Iterations(
          parse_discrete(1_UZ, L"Maximum iterations: ")) == false) {
    ERR(L"An error has been triggered from the "
        L"`Set__Maximum_Iterations()` function.");

    return false;
  }

  INFO(L"");

  if (this->Set__Use__Shuffle(accept(L"Use shuffle: ")) == false) {
    ERR(L"An error has been triggered from the "
        L"`Set__Use__Shuffle()` function.");

    return false;
  }

  INFO(L"");

  if (accept(L"Do you want to use weight decay: ")) {
    tmp_minimum = parse_real(0_r, 1_r, L"Minimum: ");
    tmp_maximum = parse_real(0_r, 1_r, L"Maximum: ");
    tmp_step_size = parse_real(0_r, 1_r, L"Step size: ");

    if (this->push_back_wd(tmp_minimum, tmp_maximum, tmp_step_size) == false) {
      ERR(L"An error has been triggered from the "
          L"`push_back_wd(%f, %f, %f)` function.",
          tmp_minimum, tmp_maximum, tmp_step_size);

      return false;
    }
  }

  INFO(L"");

  if (accept(L"Do you want to use max-norm constraints: ")) {
    tmp_minimum = parse_real(0_r, L"Minimum: ");
    tmp_maximum = parse_real(0_r, L"Maximum: ");
    tmp_step_size = parse_real(0_r, L"Step size: ");

    if (this->push_back_max_norm_constraints(tmp_minimum, tmp_maximum,
                                             tmp_step_size) == false) {
      ERR(L"An error has been triggered from the "
          L"`push_back_max_norm_constraints(%f, %f, %f)` function.",
          tmp_minimum, tmp_maximum, tmp_step_size);

      return false;
    }
  }

  INFO(L"");

  if (accept(L"Do you want to use batch-normalization momentum: ")) {
    tmp_minimum = parse_real(0_r, 1_r, L"Minimum: ");
    tmp_maximum = parse_real(0_r, 1_r, L"Maximum: ");
    tmp_step_size = parse_real(0_r, 1_r, L"Step size: ");

    if (this->push_back_normalization_mom_avg(tmp_minimum, tmp_maximum,
                                              tmp_step_size) == false) {
      ERR(L"An error has been triggered from the "
          L"`push_back_normalization_mom_avg(%f, %f, %f)` function.",
          tmp_minimum, tmp_maximum, tmp_step_size);

      return false;
    }

    if (this->push_back_normalization_mom_avg(
            1_r / static_cast<real>(ref_Neural_Network_Manager.get_datasets()
                                        ->get_dataset(ENV::TRAIN)
                                        ->get_n_batch()),
            1_r / static_cast<real>(ref_Neural_Network_Manager.get_datasets()
                                        ->get_dataset(ENV::TRAIN)
                                        ->get_n_batch()),
            1_r, false) == false) {
      ERR(L"An error has been triggered from the "
          L"`push_back_normalization_mom_avg(%f, %f, 1.0, false)` "
          L"function.",
          1_r / static_cast<real>(ref_Neural_Network_Manager.get_datasets()
                                      ->get_dataset(ENV::TRAIN)
                                      ->get_n_batch()),
          1_r / static_cast<real>(ref_Neural_Network_Manager.get_datasets()
                                      ->get_dataset(ENV::TRAIN)
                                      ->get_n_batch()));

      return false;
    }
  }

  INFO(L"");

  if (accept(L"Do you want to use dropout: ")) {
    INFO(L"");

    if (accept(L"Do you want to use dropout, bernoulli: ")) {
      if (accept(L"Do you want to use dropout at the input layer: ")) {
        Dropout_Initializer__Arguments tmp_Dropout_Initializer__Arguments;

        tmp_Dropout_Initializer__Arguments.layer_index = 0_UZ;

        tmp_Dropout_Initializer__Arguments.type_layer_dropout =
            LAYER_DROPOUT::BERNOULLI;

        tmp_Dropout_Initializer__Arguments.minval[0] =
            parse_real(0_r, 1_r, L"Minimum: ");

        tmp_Dropout_Initializer__Arguments.maxval[0] =
            parse_real(0_r, 1_r, L"Maximum: ");

        tmp_Dropout_Initializer__Arguments.step_size[0] =
            parse_real(0_r, 1_r, L"Step size: ");

        if (this->push_back_dropout(tmp_Dropout_Initializer__Arguments) ==
            false) {
          ERR(L"An error has been triggered from the "
              L"`push_back_dropout(%zu, %ls, %f, %f, %f)` function.",
              tmp_Dropout_Initializer__Arguments.layer_index,
              LAYER_DROPOUT_NAME[tmp_Dropout_Initializer__Arguments
                                     .type_layer_dropout]
                  .c_str(),
              tmp_Dropout_Initializer__Arguments.minval[0],
              tmp_Dropout_Initializer__Arguments.maxval[0],
              tmp_Dropout_Initializer__Arguments.step_size[0]);

          return false;
        }
      }

      for (tmp_layer_index = 1_UZ;
           tmp_layer_index !=
           tmp_ptr_Neural_Network->Get__Total_Layers() - 1_UZ;
           ++tmp_layer_index) {
        if (accept((L"Do you want to use dropout at the hidden layer " +
                    std::to_wstring(tmp_layer_index) + L": ")
                       .c_str())) {
          Dropout_Initializer__Arguments tmp_Dropout_Initializer__Arguments;

          tmp_Dropout_Initializer__Arguments.layer_index = tmp_layer_index;

          tmp_Dropout_Initializer__Arguments.type_layer_dropout =
              LAYER_DROPOUT::BERNOULLI;

          tmp_Dropout_Initializer__Arguments.minval[0] =
              parse_real(0_r, 1_r, L"Minimum: ");

          tmp_Dropout_Initializer__Arguments.maxval[0] =
              parse_real(0_r, 1_r, L"Maximum: ");

          tmp_Dropout_Initializer__Arguments.step_size[0] =
              parse_real(0_r, 1_r, L"Step size: ");

          if (this->push_back_dropout(tmp_Dropout_Initializer__Arguments) ==
              false) {
            ERR(L"An error has been triggered from the "
                L"`push_back_dropout(%zu, %ls, %f, %f, %f)` function.",
                tmp_Dropout_Initializer__Arguments.layer_index,
                LAYER_DROPOUT_NAME[tmp_Dropout_Initializer__Arguments
                                       .type_layer_dropout]
                    .c_str(),
                tmp_Dropout_Initializer__Arguments.minval[0],
                tmp_Dropout_Initializer__Arguments.maxval[0],
                tmp_Dropout_Initializer__Arguments.step_size[0]);

            return false;
          }
        }
      }
    }

    INFO(L"");

    if (accept(L"Do you want to use dropout, bernoulli inverted: ")) {
      if (accept(L"Do you want to use dropout at the input layer: ")) {
        Dropout_Initializer__Arguments tmp_Dropout_Initializer__Arguments;

        tmp_Dropout_Initializer__Arguments.layer_index = 0_UZ;

        tmp_Dropout_Initializer__Arguments.type_layer_dropout =
            LAYER_DROPOUT::BERNOULLI_INVERTED;

        tmp_Dropout_Initializer__Arguments.minval[0] =
            parse_real(0_r, 1_r, L"Minimum: ");

        tmp_Dropout_Initializer__Arguments.maxval[0] =
            parse_real(0_r, 1_r, L"Maximum: ");

        tmp_Dropout_Initializer__Arguments.step_size[0] =
            parse_real(0_r, 1_r, L"Step size: ");

        if (this->push_back_dropout(tmp_Dropout_Initializer__Arguments) ==
            false) {
          ERR(L"An error has been triggered from the "
              L"`push_back_dropout(%zu, %ls, %f, %f, %f)` function.",
              tmp_Dropout_Initializer__Arguments.layer_index,
              LAYER_DROPOUT_NAME[tmp_Dropout_Initializer__Arguments
                                     .type_layer_dropout]
                  .c_str(),
              tmp_Dropout_Initializer__Arguments.minval[0],
              tmp_Dropout_Initializer__Arguments.maxval[0],
              tmp_Dropout_Initializer__Arguments.step_size[0]);

          return false;
        }
      }

      for (tmp_layer_index = 1_UZ;
           tmp_layer_index !=
           tmp_ptr_Neural_Network->Get__Total_Layers() - 1_UZ;
           ++tmp_layer_index) {
        if (accept((L"Do you want to use dropout at the hidden layer " +
                    std::to_wstring(tmp_layer_index) + L": ")
                       .c_str())) {
          Dropout_Initializer__Arguments tmp_Dropout_Initializer__Arguments;

          tmp_Dropout_Initializer__Arguments.layer_index = tmp_layer_index;

          tmp_Dropout_Initializer__Arguments.type_layer_dropout =
              LAYER_DROPOUT::BERNOULLI_INVERTED;

          tmp_Dropout_Initializer__Arguments.minval[0] =
              parse_real(0_r, 1_r, L"Minimum: ");

          tmp_Dropout_Initializer__Arguments.maxval[0] =
              parse_real(0_r, 1_r, L"Maximum: ");

          tmp_Dropout_Initializer__Arguments.step_size[0] =
              parse_real(0_r, 1_r, L"Step size: ");

          if (this->push_back_dropout(tmp_Dropout_Initializer__Arguments) ==
              false) {
            ERR(L"An error has been triggered from the "
                L"`push_back_dropout(%zu, %ls, %f, %f, %f)` function.",
                tmp_Dropout_Initializer__Arguments.layer_index,
                LAYER_DROPOUT_NAME[tmp_Dropout_Initializer__Arguments
                                       .type_layer_dropout]
                    .c_str(),
                tmp_Dropout_Initializer__Arguments.minval[0],
                tmp_Dropout_Initializer__Arguments.maxval[0],
                tmp_Dropout_Initializer__Arguments.step_size[0]);

            return false;
          }
        }
      }
    }

    INFO(L"");

    if (accept(L"Do you want to use dropout, gaussian: ")) {
      if (accept(L"Do you want to use dropout at the input layer: ")) {
        Dropout_Initializer__Arguments tmp_Dropout_Initializer__Arguments;

        tmp_Dropout_Initializer__Arguments.layer_index = 0_UZ;

        tmp_Dropout_Initializer__Arguments.type_layer_dropout =
            LAYER_DROPOUT::GAUSSIAN;

        tmp_Dropout_Initializer__Arguments.minval[0] =
            parse_real(0_r, 1_r, L"Minimum: ");

        tmp_Dropout_Initializer__Arguments.maxval[0] =
            parse_real(0_r, 1_r, L"Maximum: ");

        tmp_Dropout_Initializer__Arguments.step_size[0] =
            parse_real(0_r, 1_r, L"Step size: ");

        if (this->push_back_dropout(tmp_Dropout_Initializer__Arguments) ==
            false) {
          ERR(L"An error has been triggered from the "
              L"`push_back_dropout(%zu, %ls, %f, %f, %f)` function.",
              tmp_Dropout_Initializer__Arguments.layer_index,
              LAYER_DROPOUT_NAME[tmp_Dropout_Initializer__Arguments
                                     .type_layer_dropout]
                  .c_str(),
              tmp_Dropout_Initializer__Arguments.minval[0],
              tmp_Dropout_Initializer__Arguments.maxval[0],
              tmp_Dropout_Initializer__Arguments.step_size[0]);

          return false;
        }
      }

      for (tmp_layer_index = 1_UZ;
           tmp_layer_index !=
           tmp_ptr_Neural_Network->Get__Total_Layers() - 1_UZ;
           ++tmp_layer_index) {
        if (accept((L"Do you want to use dropout at the hidden layer " +
                    std::to_wstring(tmp_layer_index) + L": ")
                       .c_str())) {
          Dropout_Initializer__Arguments tmp_Dropout_Initializer__Arguments;

          tmp_Dropout_Initializer__Arguments.layer_index = tmp_layer_index;

          tmp_Dropout_Initializer__Arguments.type_layer_dropout =
              LAYER_DROPOUT::GAUSSIAN;

          tmp_Dropout_Initializer__Arguments.minval[0] =
              parse_real(0_r, 1_r, L"Minimum: ");

          tmp_Dropout_Initializer__Arguments.maxval[0] =
              parse_real(0_r, 1_r, L"Maximum: ");

          tmp_Dropout_Initializer__Arguments.step_size[0] =
              parse_real(0_r, 1_r, L"Step size: ");

          if (this->push_back_dropout(tmp_Dropout_Initializer__Arguments) ==
              false) {
            ERR(L"An error has been triggered from the "
                L"`push_back_dropout(%zu, %ls, %f, %f, %f)` function.",
                tmp_Dropout_Initializer__Arguments.layer_index,
                LAYER_DROPOUT_NAME[tmp_Dropout_Initializer__Arguments
                                       .type_layer_dropout]
                    .c_str(),
                tmp_Dropout_Initializer__Arguments.minval[0],
                tmp_Dropout_Initializer__Arguments.maxval[0],
                tmp_Dropout_Initializer__Arguments.step_size[0]);

            return false;
          }
        }
      }
    }

    INFO(L"");

    if (accept(L"Do you want to use dropout, Uout: ")) {
      if (accept(L"Do you want to use dropout at the input layer: ")) {
        Dropout_Initializer__Arguments tmp_Dropout_Initializer__Arguments;

        tmp_Dropout_Initializer__Arguments.layer_index = 0_UZ;

        tmp_Dropout_Initializer__Arguments.type_layer_dropout =
            LAYER_DROPOUT::UOUT;

        tmp_Dropout_Initializer__Arguments.minval[0] =
            parse_real(0_r, 1_r, L"Minimum: ");

        tmp_Dropout_Initializer__Arguments.maxval[0] =
            parse_real(0_r, 1_r, L"Maximum: ");

        tmp_Dropout_Initializer__Arguments.step_size[0] =
            parse_real(0_r, 1_r, L"Step size: ");

        if (this->push_back_dropout(tmp_Dropout_Initializer__Arguments) ==
            false) {
          ERR(L"An error has been triggered from the "
              L"`push_back_dropout(%zu, %ls, %f, %f, %f)` function.",
              tmp_Dropout_Initializer__Arguments.layer_index,
              LAYER_DROPOUT_NAME[tmp_Dropout_Initializer__Arguments
                                     .type_layer_dropout]
                  .c_str(),
              tmp_Dropout_Initializer__Arguments.minval[0],
              tmp_Dropout_Initializer__Arguments.maxval[0],
              tmp_Dropout_Initializer__Arguments.step_size[0]);

          return false;
        }
      }

      for (tmp_layer_index = 1_UZ;
           tmp_layer_index !=
           tmp_ptr_Neural_Network->Get__Total_Layers() - 1_UZ;
           ++tmp_layer_index) {
        if (accept((L"Do you want to use dropout at the hidden layer " +
                    std::to_wstring(tmp_layer_index) + L": ")
                       .c_str())) {
          Dropout_Initializer__Arguments tmp_Dropout_Initializer__Arguments;

          tmp_Dropout_Initializer__Arguments.layer_index = tmp_layer_index;

          tmp_Dropout_Initializer__Arguments.type_layer_dropout =
              LAYER_DROPOUT::UOUT;

          tmp_Dropout_Initializer__Arguments.minval[0] =
              parse_real(0_r, 1_r, L"Minimum: ");

          tmp_Dropout_Initializer__Arguments.maxval[0] =
              parse_real(0_r, 1_r, L"Maximum: ");

          tmp_Dropout_Initializer__Arguments.step_size[0] =
              parse_real(0_r, 1_r, L"Step size: ");

          if (this->push_back_dropout(tmp_Dropout_Initializer__Arguments) ==
              false) {
            ERR(L"An error has been triggered from the "
                L"`push_back_dropout(%zu, %ls, %f, %f, %f)` function.",
                tmp_Dropout_Initializer__Arguments.layer_index,
                LAYER_DROPOUT_NAME[tmp_Dropout_Initializer__Arguments
                                       .type_layer_dropout]
                    .c_str(),
                tmp_Dropout_Initializer__Arguments.minval[0],
                tmp_Dropout_Initializer__Arguments.maxval[0],
                tmp_Dropout_Initializer__Arguments.step_size[0]);

            return false;
          }
        }
      }
    }

    INFO(L"");

    if (accept(L"Do you want to use dropout, Zoneout: ")) {
      if (accept(L"Do you want to use dropout at the input layer: ")) {
        Dropout_Initializer__Arguments tmp_Dropout_Initializer__Arguments;

        tmp_Dropout_Initializer__Arguments.layer_index = 0_UZ;

        tmp_Dropout_Initializer__Arguments.type_layer_dropout =
            LAYER_DROPOUT::ZONEOUT;

        tmp_Dropout_Initializer__Arguments.minval[0] =
            parse_real(0_r, 1_r, L"Minimum_0: ");

        tmp_Dropout_Initializer__Arguments.maxval[0] =
            parse_real(0_r, 1_r, L"Maximum_0: ");

        tmp_Dropout_Initializer__Arguments.step_size[0] =
            parse_real(0_r, 1_r, L"Step size_0: ");

        tmp_Dropout_Initializer__Arguments.minval[1] =
            parse_real(0_r, 1_r, L"Minimum_1: ");

        tmp_Dropout_Initializer__Arguments.maxval[1] =
            parse_real(0_r, 1_r, L"Maximum_1: ");

        tmp_Dropout_Initializer__Arguments.step_size[1] =
            parse_real(0_r, 1_r, L"Step size_1: ");

        if (this->push_back_dropout(tmp_Dropout_Initializer__Arguments) ==
            false) {
          ERR(L"An error has been triggered from the "
              L"`push_back_dropout(%zu, %ls, %f, %f, %f, %f, %f, %f)` "
              L"function.",
              tmp_Dropout_Initializer__Arguments.layer_index,
              LAYER_DROPOUT_NAME[tmp_Dropout_Initializer__Arguments
                                     .type_layer_dropout]
                  .c_str(),
              tmp_Dropout_Initializer__Arguments.minval[0],
              tmp_Dropout_Initializer__Arguments.maxval[0],
              tmp_Dropout_Initializer__Arguments.step_size[0],
              tmp_Dropout_Initializer__Arguments.minval[1],
              tmp_Dropout_Initializer__Arguments.maxval[1],
              tmp_Dropout_Initializer__Arguments.step_size[1]);

          return false;
        }
      }

      for (tmp_layer_index = 1_UZ;
           tmp_layer_index !=
           tmp_ptr_Neural_Network->Get__Total_Layers() - 1_UZ;
           ++tmp_layer_index) {
        if (accept((L"Do you want to use dropout at the hidden layer " +
                    std::to_wstring(tmp_layer_index) + L": ")
                       .c_str())) {
          Dropout_Initializer__Arguments tmp_Dropout_Initializer__Arguments;

          tmp_Dropout_Initializer__Arguments.layer_index = tmp_layer_index;

          tmp_Dropout_Initializer__Arguments.type_layer_dropout =
              LAYER_DROPOUT::ZONEOUT;

          tmp_Dropout_Initializer__Arguments.minval[0] =
              parse_real(0_r, 1_r, L"Minimum_0: ");

          tmp_Dropout_Initializer__Arguments.maxval[0] =
              parse_real(0_r, 1_r, L"Maximum_0: ");

          tmp_Dropout_Initializer__Arguments.step_size[0] =
              parse_real(0_r, 1_r, L"Step size_0: ");

          tmp_Dropout_Initializer__Arguments.minval[1] =
              parse_real(0_r, 1_r, L"Minimum_1: ");

          tmp_Dropout_Initializer__Arguments.maxval[1] =
              parse_real(0_r, 1_r, L"Maximum_1: ");

          tmp_Dropout_Initializer__Arguments.step_size[1] =
              parse_real(0_r, 1_r, L"Step size_1: ");

          if (this->push_back_dropout(tmp_Dropout_Initializer__Arguments) ==
              false) {
            ERR(L"An error has been triggered from the "
                L"`push_back_dropout(%zu, %ls, %f, %f, %f, %f, %f, %f)` "
                L"function.",
                tmp_Dropout_Initializer__Arguments.layer_index,
                LAYER_DROPOUT_NAME[tmp_Dropout_Initializer__Arguments
                                       .type_layer_dropout]
                    .c_str(),
                tmp_Dropout_Initializer__Arguments.minval[0],
                tmp_Dropout_Initializer__Arguments.maxval[0],
                tmp_Dropout_Initializer__Arguments.step_size[0],
                tmp_Dropout_Initializer__Arguments.minval[1],
                tmp_Dropout_Initializer__Arguments.maxval[1],
                tmp_Dropout_Initializer__Arguments.step_size[1]);

            return false;
          }
        }
      }
    }
  }

  return true;
}

bool Grid_Search::Set__Maximum_Iterations(
    size_t const maximum_iterations_received) {
  if (maximum_iterations_received == 0_UZ) {
    ERR(L"Maximum iterations can not be zero.");

    return false;
  }

  this->_maximum_iterations = maximum_iterations_received;

  return true;
}

bool Grid_Search::Set__Use__Shuffle(bool const use_shuffle_received) {
  this->_use_shuffle = use_shuffle_received;

  if (use_shuffle_received) {
    if (this->Allocate__Stochastic_Index() == false) {
      ERR(L"An error has been triggered from the "
          L"`Allocate__Stochastic_Index()` function.");

      return false;
    }

    this->_Generator_Random_Integer.seed(static_cast<unsigned int>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()));
  } else {
    this->Deallocate__Stochastic_Index();
  }

  return true;
}

bool Grid_Search::push_back_wd(real const minval, real const maxval,
                               real const step_size_received,
                               bool const duplicate_allowed) {
  if (minval > maxval) {
    ERR(L"Minimum value (%f) bigger than maximum value (%f).", minval, maxval);

    return false;
  } else if (minval < 0_r) {
    ERR(L"Minimum value (%f) less than zero.", minval);

    return false;
  } else if (maxval > 1_r) {
    ERR(L"Maximum value (%f) bigger than one.", maxval);

    return false;
  } else if (step_size_received == 0_r) {
    ERR(L"Step size can not be zero.");

    return false;
  }

  real tmp;

  if (minval == maxval) {
    tmp = minval;

    this->push_back(this->p_vector_Weight_Decay, tmp, duplicate_allowed);

    return true;
  }

  size_t const length(
      static_cast<size_t>((maxval - minval) / step_size_received) + 1_UZ);
  size_t i;

  // Disable.
  if (minval != 0_r) {
    tmp = 0_r;

    this->push_back(this->p_vector_Weight_Decay, tmp, duplicate_allowed);
  }

  for (i = 0_UZ; i != length; ++i) {
    tmp = minval + static_cast<real>(i) * step_size_received;

    this->push_back(this->p_vector_Weight_Decay, tmp, duplicate_allowed);
  }

  // Step size can not reach maximum.
  if (minval + static_cast<real>(length - 1_UZ) * step_size_received !=
      maxval) {
    tmp = maxval;

    this->push_back(this->p_vector_Weight_Decay, tmp, duplicate_allowed);
  }

  return true;
}

bool Grid_Search::push_back_max_norm_constraints(real const minval,
                                                 real const maxval,
                                                 real const step_size_received,
                                                 bool const duplicate_allowed) {
  if (minval > maxval) {
    ERR(L"Minimum value (%f) bigger than maximum value (%f).", minval, maxval);

    return false;
  } else if (minval < 0_r) {
    ERR(L"Minimum value (%f) less than zero.", minval);

    return false;
  } else if (step_size_received == 0_r) {
    ERR(L"Step size can not be zero.");

    return false;
  }

  real tmp;

  if (minval == maxval) {
    tmp = minval;

    this->push_back(this->p_vector_Max_Norm_Constraints, tmp,
                    duplicate_allowed);

    return true;
  }

  size_t const length(
      static_cast<size_t>((maxval - minval) / step_size_received) + 1_UZ);
  size_t i;

  // Disable.
  if (minval != 0_r) {
    tmp = 0_r;

    this->push_back(this->p_vector_Max_Norm_Constraints, tmp,
                    duplicate_allowed);
  }

  for (i = 0_UZ; i != length; ++i) {
    tmp = minval + static_cast<real>(i) * step_size_received;

    this->push_back(this->p_vector_Max_Norm_Constraints, tmp,
                    duplicate_allowed);
  }

  // Step size can not reach maximum.
  if (minval + static_cast<real>(length - 1_UZ) * step_size_received !=
      maxval) {
    tmp = maxval;

    this->push_back(this->p_vector_Max_Norm_Constraints, tmp,
                    duplicate_allowed);
  }

  return true;
}

bool Grid_Search::push_back_normalization_mom_avg(
    real const minval, real const maxval, real const step_size_received,
    bool const duplicate_allowed) {
  if (minval > maxval) {
    ERR(L"Minimum value (%f) bigger than maximum value (%f).", minval, maxval);

    return false;
  } else if (minval < 0_r) {
    ERR(L"Minimum value (%f) less than zero.", minval);

    return false;
  } else if (maxval > 1_r) {
    ERR(L"Maximum value (%f) bigger than one.", maxval);

    return false;
  } else if (step_size_received == 0_r) {
    ERR(L"Step size can not be zero.");

    return false;
  }

  real tmp;

  if (minval == maxval) {
    tmp = minval;

    this->push_back(this->p_vector_Normalization_Momentum_Average, tmp,
                    duplicate_allowed);

    return true;
  }

  size_t const length(
      static_cast<size_t>((maxval - minval) / step_size_received) + 1_UZ);
  size_t i;

  for (i = 0_UZ; i != length; ++i) {
    tmp = minval + static_cast<real>(i) * step_size_received;

    this->push_back(this->p_vector_Normalization_Momentum_Average, tmp,
                    duplicate_allowed);
  }

  // Step size can not reach maximum.
  if (minval + static_cast<real>(length - 1_UZ) * step_size_received !=
      maxval) {
    tmp = maxval;

    this->push_back(this->p_vector_Normalization_Momentum_Average, tmp,
                    duplicate_allowed);
  }

  return true;
}

bool Grid_Search::push_back_dropout(Dropout_Initializer__Arguments init_args,
                                    bool const duplicate_allowed) {
  if (init_args.minval[0] > init_args.maxval[0]) {
    ERR(L"Minimum value[0] (%f) bigger than maximum value[0] "
        L"(%f).",
        init_args.minval[0], init_args.maxval[0]);

    return false;
  } else if (init_args.minval[0] < 0_r) {
    ERR(L"Minimum value[0] (%f) less than zero.", init_args.minval[0]);

    return false;
  } else if (init_args.maxval[0] > 1_r) {
    ERR(L"Maximum value[0] (%f) bigger than one.", init_args.maxval[0]);

    return false;
  } else if (init_args.step_size[0] == 0_r) {
    ERR(L"Size_0 can not be zero.");

    return false;
  }

  switch (init_args.type_layer_dropout) {
    case LAYER_DROPOUT::ZONEOUT:
      if (init_args.minval[1] > init_args.maxval[1]) {
        ERR(L"Minimum value[1] (%f) bigger than maximum "
            L"value[1] (%f).",
            init_args.minval[1], init_args.maxval[1]);

        return false;
      } else if (init_args.minval[1] < 0_r) {
        ERR(L"Minimum value[1] (%f) less than zero.", init_args.minval[1]);

        return false;
      } else if (init_args.maxval[1] > 1_r) {
        ERR(L"Maximum value[1] (%f) bigger than one.", init_args.maxval[1]);

        return false;
      } else if (init_args.step_size[1] == 0_r) {
        ERR(L"Size_1 can not be zero.");

        return false;
      }
      break;
    default:
      if (init_args.step_size[1] >= 1_r) {
        ERR(L"Size_1 can not be greater or equal to one.");

        return false;
      }
      break;
  }

  Dropout_Initializer__LW tmp_Dropout_Initializer__LW;

  tmp_Dropout_Initializer__LW.layer_index = init_args.layer_index;

  tmp_Dropout_Initializer__LW.type_layer_dropout = init_args.type_layer_dropout;

  std::vector<Dropout_Initializer__LW> *tmp_ptr_vector_Dropout(nullptr);

  // Search if a vector contains the desired layer.
  for (auto &ref_vector : this->p_vector_layers_Dropout) {
    if (ref_vector.at(0).layer_index == init_args.layer_index) {
      tmp_ptr_vector_Dropout = &ref_vector;

      break;
    }
  }

  // If the vector can not be find. allocate a new one with the desired layer
  // index.
  if (tmp_ptr_vector_Dropout == nullptr) {
    std::vector<Dropout_Initializer__LW> tmp_vector;

    this->p_vector_layers_Dropout.push_back(tmp_vector);

    tmp_ptr_vector_Dropout = &this->p_vector_layers_Dropout.back();
  }

  if (init_args.minval[0] == init_args.maxval[0] &&
      init_args.minval[1] == init_args.maxval[1]) {
    tmp_Dropout_Initializer__LW.value[0] = init_args.minval[0];

    tmp_Dropout_Initializer__LW.value[1] = init_args.minval[1];

    this->push_back(*tmp_ptr_vector_Dropout, tmp_Dropout_Initializer__LW,
                    duplicate_allowed);

    return true;
  }

  size_t length, tmp_sub_size, i, tmp_sub_index;

  switch (init_args.type_layer_dropout) {
    case LAYER_DROPOUT::BERNOULLI:
    case LAYER_DROPOUT::BERNOULLI_INVERTED:
      break;
    case LAYER_DROPOUT::GAUSSIAN:
    case LAYER_DROPOUT::UOUT:
    case LAYER_DROPOUT::ZONEOUT:
      // Disable.
      if (init_args.minval[0] != 0_r || init_args.minval[1] != 0_r) {
        tmp_Dropout_Initializer__LW.value[0] = 0_r;

        tmp_Dropout_Initializer__LW.value[1] = 0_r;

        this->push_back(*tmp_ptr_vector_Dropout, tmp_Dropout_Initializer__LW,
                        false);
      }
      break;
    default:
      ERR(L"Layer type dropout (%d | %ls) is not managed in the "
          L"switch.",
          init_args.type_layer_dropout,
          LAYER_DROPOUT_NAME[init_args.type_layer_dropout].c_str());
      return false;
  }

  // Value[0].
  if (init_args.step_size[0] != 0_r) {
    length = static_cast<size_t>((init_args.maxval[0] - init_args.minval[0]) /
                                 init_args.step_size[0]) +
             1_UZ;

    tmp_Dropout_Initializer__LW.value[1] = 0_r;

    for (i = 0_UZ; i != length; ++i) {
      tmp_Dropout_Initializer__LW.value[0] =
          init_args.minval[0] + static_cast<real>(i) * init_args.step_size[0];

      this->push_back(*tmp_ptr_vector_Dropout, tmp_Dropout_Initializer__LW,
                      duplicate_allowed);
    }

    // Step size can not reach maximum.
    if (tmp_Dropout_Initializer__LW.value[0] != init_args.maxval[0]) {
      tmp_Dropout_Initializer__LW.value[0] = init_args.maxval[0];

      this->push_back(*tmp_ptr_vector_Dropout, tmp_Dropout_Initializer__LW,
                      false);
    }
  }
  // |END| Value[0]. |END|

  // Value[1].
  if (init_args.step_size[1] != 0_r) {
    length = static_cast<size_t>((init_args.maxval[1] - init_args.minval[1]) /
                                 init_args.step_size[1]) +
             1_UZ;

    tmp_Dropout_Initializer__LW.value[0] = 0_r;

    for (i = 0_UZ; i != length; ++i) {
      tmp_Dropout_Initializer__LW.value[1] =
          init_args.minval[1] + static_cast<real>(i) * init_args.step_size[1];

      this->push_back(*tmp_ptr_vector_Dropout, tmp_Dropout_Initializer__LW,
                      duplicate_allowed);
    }

    // Step size can not reach maximum.
    if (tmp_Dropout_Initializer__LW.value[1] != init_args.maxval[1]) {
      tmp_Dropout_Initializer__LW.value[1] = init_args.maxval[1];

      this->push_back(*tmp_ptr_vector_Dropout, tmp_Dropout_Initializer__LW,
                      false);
    }
  }
  // |END| Value[1]. |END|

  // Value[0] && Value[1].
  if (init_args.step_size[0] * init_args.step_size[1] != 0_r) {
    length = static_cast<size_t>((init_args.maxval[0] - init_args.minval[0]) /
                                 init_args.step_size[0]) +
             1_UZ;
    tmp_sub_size =
        static_cast<size_t>((init_args.maxval[1] - init_args.minval[1]) /
                            init_args.step_size[1]) +
        1_UZ;

    for (i = 0_UZ; i != length; ++i) {
      for (tmp_sub_index = 0_UZ; tmp_sub_index != tmp_sub_size;
           ++tmp_sub_index) {
        tmp_Dropout_Initializer__LW.value[0] =
            init_args.minval[0] + static_cast<real>(i) * init_args.step_size[0];
        tmp_Dropout_Initializer__LW.value[1] =
            init_args.minval[1] +
            static_cast<real>(tmp_sub_index) * init_args.step_size[1];

        this->push_back(*tmp_ptr_vector_Dropout, tmp_Dropout_Initializer__LW,
                        duplicate_allowed);
      }
    }
  }
  // |END| Value[0] && Value[1]. |END|

  switch (init_args.type_layer_dropout) {
    case LAYER_DROPOUT::BERNOULLI:
    case LAYER_DROPOUT::BERNOULLI_INVERTED:
      // Disable.
      if (init_args.maxval[0] != 1_r) {
        tmp_Dropout_Initializer__LW.value[0] = 1_r;

        this->push_back(*tmp_ptr_vector_Dropout, tmp_Dropout_Initializer__LW,
                        false);
      }
      break;
    case LAYER_DROPOUT::GAUSSIAN:
    case LAYER_DROPOUT::UOUT:
    case LAYER_DROPOUT::ZONEOUT:
      break;
    default:
      ERR(L"Layer type dropout (%d | %ls) is not managed in the "
          L"switch.",
          init_args.type_layer_dropout,
          LAYER_DROPOUT_NAME[init_args.type_layer_dropout].c_str());
      return false;
  }

  return true;
}

bool Grid_Search::update_tree(void) {
  size_t tmp_vector_size, tmp_vector_depth_index, tmp_vector_depth_size;

  this->_vector_Tree.clear();

  tmp_vector_depth_size = 0_UZ;

  if (this->p_vector_Weight_Decay.empty() == false) {
    this->_vector_Tree.push_back(1_UZ);

    this->_vector_Tree.at(tmp_vector_depth_size) =
        this->p_vector_Weight_Decay.size();

    ++tmp_vector_depth_size;
  }

  if (this->p_vector_Max_Norm_Constraints.empty() == false) {
    this->_vector_Tree.push_back(1_UZ);

    this->_vector_Tree.at(tmp_vector_depth_size) =
        this->p_vector_Max_Norm_Constraints.size();

    ++tmp_vector_depth_size;
  }

  if (this->p_vector_Normalization_Momentum_Average.empty() == false) {
    this->_vector_Tree.push_back(1_UZ);

    this->_vector_Tree.at(tmp_vector_depth_size) =
        this->p_vector_Normalization_Momentum_Average.size();

    ++tmp_vector_depth_size;
  }

  if (this->p_vector_layers_Dropout.empty() == false) {
    tmp_vector_size = this->p_vector_layers_Dropout.size();

    for (tmp_vector_depth_index = 0_UZ;
         tmp_vector_depth_index != tmp_vector_size; ++tmp_vector_depth_index) {
      if (this->p_vector_layers_Dropout.at(tmp_vector_depth_index).empty() ==
          false) {
        this->_vector_Tree.push_back(1_UZ);

        this->_vector_Tree.at(tmp_vector_depth_size) =
            this->p_vector_layers_Dropout.at(tmp_vector_depth_index).size();

        ++tmp_vector_depth_size;
      }
    }
  }

  if (tmp_vector_depth_size == 0_UZ) {
    ERR(L"Tree is empty.");

    return false;
  }

  size_t const tmp_trunc_size(this->_vector_Tree.at(0));

  // Loop in each vectors.
  for (tmp_vector_depth_index = 0_UZ;
       tmp_vector_depth_index != tmp_vector_depth_size - 1_UZ;
       ++tmp_vector_depth_index) {
    // Self.
    this->_vector_Tree.at(tmp_vector_depth_index) = 1_UZ;

    // Multiplacate path.
    this->_vector_Tree.at(tmp_vector_depth_index) +=
        Math::recursive_fused_multiply_add(this->_vector_Tree.data(),
                                           tmp_vector_depth_index + 1_UZ,
                                           tmp_vector_depth_size - 1_UZ);
  }

  // Last vector initialized at one.
  this->_vector_Tree.at(tmp_vector_depth_index) = 1_UZ;

  if (this->_use_shuffle) {
    if (this->Reallocate__Stochastic_Index(
            tmp_trunc_size * this->_vector_Tree.at(0)) == false) {
      ERR(L"An error has been triggered from the "
          L"`Reallocate__Stochastic_Index(%zu)` function.",
          tmp_trunc_size * this->_vector_Tree.at(0));

      return false;
    }
  }

  this->_total_iterations = tmp_trunc_size * this->_vector_Tree.at(0);

  return true;
}

bool Grid_Search::user_controls(void) {
  while (true) {
    INFO(L"");
    INFO(L"User controls:");
    INFO(L"[0]: Maximum iterations (%zu).", this->_maximum_iterations);
    INFO(L"[1]: Use shuffle (%ls).", to_wstring(this->_use_shuffle).c_str());
    INFO(L"[2]: Quit.");

    switch (parse_discrete(0, 2, L"Option: ")) {
      case 0:
        if (this->Set__Maximum_Iterations(parse_discrete(
                1_UZ, L"Maximum iterations: ")) == false) {
          ERR(L"An error has been triggered from the "
              L"`Set__Maximum_Iterations()` function.");

          return false;
        }
        break;
      case 1:
        if (this->User_Controls__Shuffle() == false) {
          ERR(L"An error has been triggered from the "
              L"`User_Controls__Shuffle()` function.");

          return false;
        }
        break;
      case 2:
        return true;
      default:
        ERR(L"An error has been triggered from the "
            L"`parse_discrete(%d, %d)` function.",
            0, 2);
        break;
    }
  }

  return true;
}

bool Grid_Search::User_Controls__Shuffle(void) {
  bool const tmp_use_shuffle(this->_use_shuffle);

  if (this->Set__Use__Shuffle(accept(L"Use shuffle: ")) == false) {
    ERR(L"An error has been triggered from the "
        L"`Set__Use__Shuffle()` function.");

    return false;
  }

  if (tmp_use_shuffle == false && this->_use_shuffle) {
    this->Shuffle();
  }

  return true;
}

bool Grid_Search::optimize(Models &models) {
  if (this->_use_shuffle) {
    this->Shuffle();
  }

  WhileCond while_cond;

  while_cond.type = WHILE_MODE::ITERATION;
  while_cond.maximum_iterations = this->_maximum_iterations;

  models.set_while_cond(while_cond);

  models.Set__Comparison_Expiration(0u);

  class Model *const tmp_ptr_Neural_Network_trainer(
      models.get_model(HIERARCHY::TRAINER)),
      *const tmp_ptr_Neural_Network_competitor(
          models.get_model(HIERARCHY::COMPETITOR)),
      *tmp_ptr_Neural_Network_trainer_clone(nullptr);

  size_t tmp_hyper_parameters_iteration(0u);

  class Keyboard keyboard;

  if ((tmp_ptr_Neural_Network_trainer_clone = new class Model) == nullptr) {
    ERR(L"Can not allocate %zu bytes.", sizeof(class Model));

    return false;
  }

  if (tmp_ptr_Neural_Network_trainer_clone->copy(
          *tmp_ptr_Neural_Network_competitor, false) == false) {
    ERR(L"An error has been triggered from the `copy(*ptr)` "
        L"function.");

    delete (tmp_ptr_Neural_Network_trainer_clone);

    return false;
  }

  while (true) {
#ifdef _WIN32
    if (keyboard.trigger_key(0x45)) {
      INFO(
          L"A signal for stopping the training has been triggered from the "
          L"user input.");

      break;
    } else if (keyboard.trigger_key(0x50)) {
      if (this->user_controls() == false) {
        ERR(L"An error has been triggered from the "
            L"`user_controls()` function.");
      }

      if (while_cond.maximum_iterations != this->_maximum_iterations) {
        while_cond.maximum_iterations = this->_maximum_iterations;

        if (models.set_while_cond(while_cond) == false) {
          ERR(L"An error has been triggered from the "
              L"`set_while_cond(ref)` function.");
        }
      }
    }
#elif __linux__
    keyboard.collect_keys_pressed();

    if (keyboard.trigger_key('q')) {
      INFO(
          L"A signal for stopping the training has been triggered from the "
          L"user input.");

      keyboard.clear_keys_pressed();

      break;
    } else if (keyboard.trigger_key('m')) {
      keyboard.clear_keys_pressed();

      if (this->user_controls() == false) {
        ERR(L"An error has been triggered from the "
            L"`user_controls()` function.");
      }

      if (while_cond.maximum_iterations != this->_maximum_iterations) {
        while_cond.maximum_iterations = this->_maximum_iterations;

        if (models.set_while_cond(while_cond) == false) {
          ERR(L"An error has been triggered from the "
              L"`set_while_cond(ref)` function.");
        }
      }
    }
#endif

    if (tmp_ptr_Neural_Network_trainer->Update(
            *tmp_ptr_Neural_Network_trainer_clone, true) == false) {
      ERR(L"An error has been triggered from the `Update(*ptr, "
          L"true)` function.");

      delete (tmp_ptr_Neural_Network_trainer_clone);

      return false;
    }

    INFO(L"");
    INFO(L"[%zu]: Feed hyper parameters.", tmp_hyper_parameters_iteration);
    if (this->Feed_Hyper_Parameters(this->_use_shuffle
                                        ? this->_ptr_array_stochastic_index
                                              [tmp_hyper_parameters_iteration]
                                        : tmp_hyper_parameters_iteration,
                                    tmp_ptr_Neural_Network_trainer) == false) {
      break;
    }

    if (tmp_ptr_Neural_Network_trainer->ptr_array_derivatives_parameters !=
        nullptr) {
      tmp_ptr_Neural_Network_trainer->clear_training_arrays();
    }

    INFO(L"Search grid, optimization [%.2f%%] %zu / %zu.",
         100.0 * static_cast<double>(tmp_hyper_parameters_iteration) /
             static_cast<double>(this->_total_iterations),
         tmp_hyper_parameters_iteration, this->_total_iterations);
    models.optimize();

    if (Sys::shutdownblock->preparing_for_shutdown()) {
      break;
    }

    ++tmp_hyper_parameters_iteration;
  }

  delete (tmp_ptr_Neural_Network_trainer_clone);

  INFO(L"");
  INFO(L"Search grid, finish [%.2f%%] %zu / %zu.",
       100.0 * static_cast<double>(tmp_hyper_parameters_iteration) /
           static_cast<double>(this->_total_iterations),
       tmp_hyper_parameters_iteration, this->_total_iterations);

  return true;
}

bool Grid_Search::Feed_Hyper_Parameters(
    size_t const hyper_parameters_index_received, class Model *const model) {
  size_t tmp_vector_size, tmp_vector_depth_index;

  size_t tmp_depth_level(0), tmp_depth_level_shift(1),
      tmp_vector_hyper_parameters_index(hyper_parameters_index_received);

  if (this->p_vector_Weight_Decay.empty() == false) {
    // Global index to vector index.
    tmp_vector_hyper_parameters_index = hyper_parameters_index_received;

    // Normalize.
    if (tmp_depth_level + 1_UZ != this->_vector_Tree.size()) {
      tmp_vector_hyper_parameters_index /=
          this->_vector_Tree.at(tmp_depth_level);

      tmp_depth_level_shift = tmp_vector_hyper_parameters_index *
                              this->_vector_Tree.at(tmp_depth_level);

      ++tmp_depth_level;
    }

    // Overflow.
    if (tmp_vector_hyper_parameters_index >=
        this->p_vector_Weight_Decay.size()) {
      return false;
    }

    INFO(L"[%zu]: Weight_Decay(%zu): %f", hyper_parameters_index_received,
         tmp_vector_hyper_parameters_index,
         this->p_vector_Weight_Decay.at(tmp_vector_hyper_parameters_index));
    if (model->set_weight_decay(this->p_vector_Weight_Decay.at(
            tmp_vector_hyper_parameters_index)) == false) {
      ERR(L"An error has been triggered from the "
          L"`set_weight_decay(%f)` function.",
          this->p_vector_Weight_Decay.at(tmp_vector_hyper_parameters_index));

      return false;
    }
  }

  if (this->p_vector_Max_Norm_Constraints.empty() == false) {
    // Depth overflow.
    if (hyper_parameters_index_received <
        tmp_depth_level_shift + tmp_depth_level) {
      return true;
    }

    // Global index to vector index.
    tmp_vector_hyper_parameters_index =
        hyper_parameters_index_received -
        (tmp_depth_level_shift + tmp_depth_level);

    // Normalize.
    if (tmp_depth_level + 1_UZ != this->_vector_Tree.size()) {
      tmp_vector_hyper_parameters_index /=
          this->_vector_Tree.at(tmp_depth_level);

      tmp_depth_level_shift += tmp_vector_hyper_parameters_index *
                               this->_vector_Tree.at(tmp_depth_level);

      ++tmp_depth_level;
    }

    // Vector overflow.
    if (tmp_vector_hyper_parameters_index >=
        this->p_vector_Max_Norm_Constraints.size()) {
      return false;
    }

    INFO(L"[%zu]: Max_Norm_Constraints(%zu): %f",
         hyper_parameters_index_received, tmp_vector_hyper_parameters_index,
         this->p_vector_Max_Norm_Constraints.at(
             tmp_vector_hyper_parameters_index));
    if (model->Set__Regularization__Max_Norm_Constraints(
            this->p_vector_Max_Norm_Constraints.at(
                tmp_vector_hyper_parameters_index)) == false) {
      ERR(L"An error has been triggered from the "
          L"`Set__Regularization__Max_Norm_Constraints(%f)` function.",
          this->p_vector_Max_Norm_Constraints.at(
              tmp_vector_hyper_parameters_index));

      return false;
    }
  }

  if (this->p_vector_Normalization_Momentum_Average.empty() == false) {
    // Depth overflow.
    if (hyper_parameters_index_received <
        tmp_depth_level_shift + tmp_depth_level) {
      return true;
    }

    // Global index to vector index.
    tmp_vector_hyper_parameters_index =
        hyper_parameters_index_received -
        (tmp_depth_level_shift + tmp_depth_level);

    // Normalize.
    if (tmp_depth_level + 1_UZ != this->_vector_Tree.size()) {
      tmp_vector_hyper_parameters_index /=
          this->_vector_Tree.at(tmp_depth_level);

      tmp_depth_level_shift += tmp_vector_hyper_parameters_index *
                               this->_vector_Tree.at(tmp_depth_level);

      ++tmp_depth_level;
    }

    // Vector overflow.
    if (tmp_vector_hyper_parameters_index >=
        this->p_vector_Normalization_Momentum_Average.size()) {
      return false;
    }

    INFO(L"[%zu]: Normalization_Momentum_Average(%d): %f",
         hyper_parameters_index_received, tmp_vector_hyper_parameters_index,
         this->p_vector_Normalization_Momentum_Average.at(
             tmp_vector_hyper_parameters_index));
    if (model->Set__Normalization_Momentum_Average(
            this->p_vector_Normalization_Momentum_Average.at(
                tmp_vector_hyper_parameters_index)) == false) {
      ERR(L"An error has been triggered from the "
          L"`Set__Normalization_Momentum_Average(%f)` function.",
          this->p_vector_Normalization_Momentum_Average.at(
              tmp_vector_hyper_parameters_index));

      return false;
    }
  }

  if (this->p_vector_layers_Dropout.empty() == false) {
    tmp_vector_size = this->p_vector_layers_Dropout.size();

    for (tmp_vector_depth_index = 0_UZ;
         tmp_vector_depth_index != tmp_vector_size; ++tmp_vector_depth_index) {
      if (this->p_vector_layers_Dropout.at(tmp_vector_depth_index).empty() ==
          false) {
        // Depth overflow.
        if (hyper_parameters_index_received <
            tmp_depth_level_shift + tmp_depth_level) {
          return true;
        }

        // Global index to vector index.
        tmp_vector_hyper_parameters_index =
            hyper_parameters_index_received -
            (tmp_depth_level_shift + tmp_depth_level);

        // Normalize.
        if (tmp_depth_level + 1_UZ != this->_vector_Tree.size()) {
          tmp_vector_hyper_parameters_index /=
              this->_vector_Tree.at(tmp_depth_level);

          tmp_depth_level_shift += tmp_vector_hyper_parameters_index *
                                   this->_vector_Tree.at(tmp_depth_level);

          ++tmp_depth_level;
        }

        // Vector overflow.
        if (tmp_vector_hyper_parameters_index >=
            this->p_vector_layers_Dropout.at(tmp_vector_depth_index).size()) {
          return false;
        }

        INFO(
            L"[%zu]: Dropout(%zu): Layer(%zu), Type(d | %ls), Value[0](%f), "
            L"Value[1](%f)",
            hyper_parameters_index_received, tmp_vector_hyper_parameters_index,
            this->p_vector_layers_Dropout.at(tmp_vector_depth_index)
                .at(tmp_vector_hyper_parameters_index)
                .layer_index,
            this->p_vector_layers_Dropout.at(tmp_vector_depth_index)
                .at(tmp_vector_hyper_parameters_index)
                .type_layer_dropout,
            LAYER_DROPOUT_NAME[this->p_vector_layers_Dropout
                                   .at(tmp_vector_depth_index)
                                   .at(tmp_vector_hyper_parameters_index)
                                   .type_layer_dropout]
                .c_str(),
            this->p_vector_layers_Dropout.at(tmp_vector_depth_index)
                .at(tmp_vector_hyper_parameters_index)
                .value[0],
            this->p_vector_layers_Dropout.at(tmp_vector_depth_index)
                .at(tmp_vector_hyper_parameters_index)
                .value[1]);
        if (model->set_dropout(
                this->p_vector_layers_Dropout.at(tmp_vector_depth_index)
                    .at(tmp_vector_hyper_parameters_index)
                    .layer_index,
                this->p_vector_layers_Dropout.at(tmp_vector_depth_index)
                    .at(tmp_vector_hyper_parameters_index)
                    .type_layer_dropout,
                std::array<real, 2_UZ>{
                    this->p_vector_layers_Dropout.at(tmp_vector_depth_index)
                        .at(tmp_vector_hyper_parameters_index)
                        .value[0],
                    this->p_vector_layers_Dropout.at(tmp_vector_depth_index)
                        .at(tmp_vector_hyper_parameters_index)
                        .value[1]}
                    .data(),
                true) == false) {
          ERR(L"An error has been triggered from the "
              L"`set_dropout(%zu, %d | %ls, %f, %f, true)` function.",
              this->p_vector_layers_Dropout.at(tmp_vector_depth_index)
                  .at(tmp_vector_hyper_parameters_index)
                  .layer_index,
              this->p_vector_layers_Dropout.at(tmp_vector_depth_index)
                  .at(tmp_vector_hyper_parameters_index)
                  .type_layer_dropout,
              LAYER_DROPOUT_NAME[this->p_vector_layers_Dropout
                                     .at(tmp_vector_depth_index)
                                     .at(tmp_vector_hyper_parameters_index)
                                     .type_layer_dropout]
                  .c_str(),
              this->p_vector_layers_Dropout.at(tmp_vector_depth_index)
                  .at(tmp_vector_hyper_parameters_index)
                  .value[0],
              this->p_vector_layers_Dropout.at(tmp_vector_depth_index)
                  .at(tmp_vector_hyper_parameters_index)
                  .value[1]);

          return false;
        }
      }
    }
  }

  return tmp_depth_level != 0_UZ;
}

template <typename T>
void Grid_Search::push_back(std::vector<T> &vec, T &obj,
                            bool const duplicate_allowed) {
  if (duplicate_allowed)
    vec.push_back(obj);
  else if (std::find(vec.begin(), vec.end(), obj) == vec.end())
    vec.push_back(obj);
}

void Grid_Search::Deallocate__Stochastic_Index(void) {
  SAFE_DELETE(this->_ptr_array_stochastic_index);
}

bool Grid_Search::Allocate__Stochastic_Index(void) {
  size_t const tmp_stochastic_index_size(
      this->_total_iterations == 0_UZ ? 1_UZ : this->_total_iterations);

  this->_ptr_array_stochastic_index = new size_t[tmp_stochastic_index_size];
  if (this->_ptr_array_stochastic_index == nullptr) {
    ERR(L"Can not allocate %zu bytes.",
        tmp_stochastic_index_size * sizeof(size_t));

    return false;
  }

  for (size_t i(0_UZ); i != tmp_stochastic_index_size; ++i)
    this->_ptr_array_stochastic_index[i] = i;

  return true;
}

bool Grid_Search::Reallocate__Stochastic_Index(
    size_t const total_iterations_received) {
  this->_ptr_array_stochastic_index = Mem::reallocate<size_t, false>(
      this->_ptr_array_stochastic_index, total_iterations_received,
      this->_total_iterations);
  if (this->_ptr_array_stochastic_index == nullptr) {
    ERR(L"Can not allocate %zu bytes.",
        total_iterations_received * sizeof(size_t));

    return false;
  }

  for (size_t i(0_UZ); i != total_iterations_received; ++i)
    this->_ptr_array_stochastic_index[i] = i;

  return true;
}
}  // namespace DL