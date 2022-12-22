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
#include "deep-learning/io/logger.hpp"
#include "deep-learning/io/term/input.hpp"
#include "deep-learning/v1/mem/reallocate.hpp"

#include <iostream>
#include <array>
#include <omp.h>

using namespace DL::Str;
using namespace DL::Term;
using namespace DL::v1::Mem;

namespace DL::v1 {
Gaussian_Search::Gaussian_Search(void) {}

bool Gaussian_Search::Initialize__OpenMP(void) {
  if (this->_is_mp_initialized == false) {
    this->_is_mp_initialized = true;

    if (this->update_mem_thread_size(this->_population_size) == false) {
      ERR(L"An error has been triggered from the "
          L"`update_mem_thread_size(%zu)` function.",
          this->_population_size);

      return false;
    }

    omp_set_dynamic(0);
  }

  return true;
}

bool Gaussian_Search::set_mp(bool const use_openmp_received) {
  if ((this->_use_mp == false && use_openmp_received) ||
      (this->_use_mp && use_openmp_received &&
       this->_is_mp_initialized == false)) {
    if (this->Initialize__OpenMP() == false) {
      ERR(L"An error has been triggered from the "
          L"`Initialize__OpenMP()` function.");

      return false;
    }
  } else if ((this->_use_mp && use_openmp_received == false) ||
             (this->_use_mp == false && use_openmp_received == false &&
              this->_is_mp_initialized)) {
    if (this->Deinitialize__OpenMP() == false) {
      ERR(L"An error has been triggered from the "
          L"`Deinitialize__OpenMP()` function.");

      return false;
    }
  }

  this->_use_mp = use_openmp_received;

  return true;
}

bool Gaussian_Search::Set__Population_Size(
    size_t const population_size_received) {
  if (population_size_received == 0_UZ) {
    ERR(L"The population size can not be equal to zero.");

    return false;
  }

  if (this->_population_size != population_size_received) {
    if (this->_population_size == 0_UZ) {
      if (this->Allocate__Population(population_size_received) == false) {
        ERR(L"An error has been triggered from the "
            L"`Allocate__Population(%zu)` function.",
            population_size_received);

        return false;
      }
    } else {
      if (this->Reallocate__Population(population_size_received) == false) {
        ERR(L"An error has been triggered from the "
            L"`Reallocate__Population(%zu)` function.",
            population_size_received);

        return false;
      }
    }

    this->_population_size = population_size_received;

    if (this->update_mem_thread_size(population_size_received) == false) {
      ERR(L"An error has been triggered from the "
          L"`update_mem_thread_size(%zu)` function.",
          population_size_received);

      return false;
    }
  }

  return true;
}

bool Gaussian_Search::Set__Population_Gaussian(
    double const population_gaussian_percent_received) {
  if (population_gaussian_percent_received <= 1.0) {
    ERR(L"The population gaussian in percent can not be equal or "
        L"less than one percent.");

    return false;
  }

  this->_population_gaussian_percent = population_gaussian_percent_received;

  return true;
}

bool Gaussian_Search::Set__Maximum_Thread_Usage(
    double const percentage_maximum_thread_usage_received) {
  if (this->_percentage_maximum_thread_usage ==
      percentage_maximum_thread_usage_received) {
    return true;
  }

  this->_percentage_maximum_thread_usage =
      percentage_maximum_thread_usage_received;

  if (this->update_mem_thread_size(this->_population_size) == false) {
    ERR(L"An error has been triggered from the "
        L"`update_mem_thread_size(%zu)` function.",
        this->_population_size);

    return false;
  }

  return true;
}

bool Gaussian_Search::Allocate__Population(
    size_t const population_size_received) {
  if (this->individuals == nullptr &&
      this->p_ptr_array_individuals == nullptr) {
    this->individuals = new Model *[population_size_received];

    if (population_size_received > 1_UZ) {
      this->p_ptr_array_individuals =
          new Model[population_size_received - 1_UZ];

      for (size_t k(1_UZ);
           k != population_size_received;
           ++k) {
        this->individuals[k] =
            this->p_ptr_array_individuals + (k - 1_UZ);
      }
    }
  }

  return true;
}

bool Gaussian_Search::Allocate__Thread(size_t const number_threads_received) {
  if (this->p_ptr_array_ptr_dataset_manager == nullptr &&
      this->p_ptr_array_dataset_manager == nullptr) {
    this->p_ptr_array_ptr_dataset_manager =
        new Datasets *[number_threads_received];

    if (number_threads_received > 1_UZ) {
      this->p_ptr_array_dataset_manager =
          new Datasets[number_threads_received - 1_UZ];

      for (size_t tmp_thread_index(1_UZ);
           tmp_thread_index != number_threads_received; ++tmp_thread_index) {
        this->p_ptr_array_ptr_dataset_manager[tmp_thread_index] =
            this->p_ptr_array_dataset_manager + (tmp_thread_index - 1_UZ);
      }
    }
  }

  return true;
}

bool Gaussian_Search::Reallocate__Population(
    size_t const population_size_received) {
  if (this->individuals != nullptr &&
      this->p_ptr_array_individuals != nullptr) {
    this->individuals = reallocate_ptofpt<Model *, false>(
        this->individuals, population_size_received,
        this->_population_size);

    if (population_size_received > 1_UZ) {
      this->p_ptr_array_individuals = reallocate_obj<Model, false>(
          this->p_ptr_array_individuals, population_size_received - 1_UZ,
          this->_population_size - 1_UZ);

      for (size_t k(1_UZ);
           k != population_size_received;
           ++k) {
        this->individuals[k] =
            this->p_ptr_array_individuals + (k - 1_UZ);
      }
    }
  }

  return true;
}

bool Gaussian_Search::Reallocate__Thread(size_t const number_threads_received) {
  if (this->p_ptr_array_ptr_dataset_manager != nullptr &&
      this->p_ptr_array_dataset_manager != nullptr) {
    this->p_ptr_array_ptr_dataset_manager =
        reallocate_ptofpt<Datasets *, false>(
            this->p_ptr_array_ptr_dataset_manager, number_threads_received,
            this->_number_threads);

    if (number_threads_received > 1_UZ) {
      this->p_ptr_array_dataset_manager = reallocate_obj<Datasets, false>(
          this->p_ptr_array_dataset_manager, number_threads_received - 1_UZ,
          this->_number_threads - 1_UZ);

      for (size_t tmp_thread_index(1_UZ);
           tmp_thread_index != number_threads_received; ++tmp_thread_index) {
        this->p_ptr_array_ptr_dataset_manager[tmp_thread_index] =
            this->p_ptr_array_dataset_manager + (tmp_thread_index - 1_UZ);
      }
    }
  }

  return true;
}

bool Gaussian_Search::Allouable__Thread_Size(
    size_t const desired_number_threads_received,
    size_t &ref_number_threads_allouable_received) {
  // TODO: Available memory.
  size_t const tmp_maximum_threads(
      (this->_use_mp || this->_is_mp_initialized)
          ? (this->_percentage_maximum_thread_usage == 0.0
                 ? 1_UZ
                 : std::min<size_t>(
                       static_cast<size_t>(ceil(
                           this->_percentage_maximum_thread_usage *
                           static_cast<double>(omp_get_num_procs()) / 100.0)),
                       desired_number_threads_received))
          : 1_UZ);
  size_t tmp_threads_allocate(
      std::min<size_t>(desired_number_threads_received, tmp_maximum_threads));

  ref_number_threads_allouable_received = tmp_threads_allocate;

  return true;
}

bool Gaussian_Search::update_mem_thread_size(
    size_t const desired_number_threads_received) {
  if (this->_is_mp_initialized == false) {
    return true;
  } else if (desired_number_threads_received <= this->_cache_number_threads &&
             this->_percentage_maximum_thread_usage ==
                 this->_cache_maximum_threads_percent) {
    return true;
  }

  size_t tmp_number_threads_allocate(desired_number_threads_received);

  if (this->Allouable__Thread_Size(desired_number_threads_received,
                                   tmp_number_threads_allocate) == false) {
    ERR(L"An error has been triggered from the "
        L"`Allouable__Thread_Size(%zu, %zu)` function.",
        desired_number_threads_received, tmp_number_threads_allocate);

    return false;
  }

  // If number of threads differ from the new desired.
  if (this->_number_threads != tmp_number_threads_allocate) {
    if (this->_number_threads == 0_UZ) {
      if (this->Allocate__Thread(tmp_number_threads_allocate) == false) {
        ERR(L"An error has been triggered from the "
            L"`Allocate__Thread(%zu)` function.",
            tmp_number_threads_allocate);

        return false;
      }
    } else {
      if (this->Reallocate__Thread(tmp_number_threads_allocate) == false) {
        ERR(L"An error has been triggered from the "
            L"`Reallocate__Thread(%zu)` function.",
            tmp_number_threads_allocate);

        return false;
      }
    }

    this->_number_threads = tmp_number_threads_allocate;
  }

  // If number of threads is greater than the cached.
  if (desired_number_threads_received > this->_cache_number_threads) {
    this->_cache_number_threads = desired_number_threads_received;
  }

  // Cache the maximum threads in percent.
  this->_cache_maximum_threads_percent = this->_percentage_maximum_thread_usage;

  return true;
}

bool Gaussian_Search::Update__Thread_Size__Population(
    size_t const desired_number_threads_received) {
  if (this->individuals[0]->use_mp &&
      this->individuals[0]->is_mp_initialized) {
    Model *indi;

    for (size_t k(1_UZ);
         k != this->_population_size;
         ++k) {
      indi =
          this->individuals[k];

      if (indi->update_mem_thread_size(
              desired_number_threads_received) == false) {
        ERR(L"An error has been triggered from the "
            L"`update_mem_thread_size(%zu)` function.",
            desired_number_threads_received);

        return false;
      }
    }
  }

  return true;
}

bool Gaussian_Search::Update__Batch_Size__Population(
    size_t const desired_batch_size_received) {
  Model *indi;

  for (size_t k(1_UZ);
       k != this->_population_size; ++k) {
    indi =
        this->individuals[k];

    if (indi->update_mem_batch_size(
            desired_batch_size_received) == false) {
      ERR(L"An error has been triggered from the "
          L"`update_mem_batch_size(%zu)` function.",
          desired_batch_size_received);

      return false;
    }
  }

  return true;
}

bool Gaussian_Search::Update__Population(
    Model *const ptr_source_Dataset_Manager_received) {
  this->individuals[0] = ptr_source_Dataset_Manager_received;

  Model *indi;

  for (size_t k(1_UZ);
       k != this->_population_size; ++k) {
    indi =
        this->individuals[k];

    if (indi->type !=
            ptr_source_Dataset_Manager_received->type ||
        indi->total_basic_units !=
            ptr_source_Dataset_Manager_received->total_basic_units ||
        indi->total_basic_indice_units !=
            ptr_source_Dataset_Manager_received->total_basic_indice_units ||
        indi->total_neuron_units !=
            ptr_source_Dataset_Manager_received->total_neuron_units ||
        indi->total_AF_units !=
            ptr_source_Dataset_Manager_received->total_AF_units ||
        indi->total_AF_Ind_recurrent_units !=
            ptr_source_Dataset_Manager_received->total_AF_Ind_recurrent_units ||
        indi->total_block_units !=
            ptr_source_Dataset_Manager_received->total_block_units ||
        indi->total_cell_units !=
            ptr_source_Dataset_Manager_received->total_cell_units ||
        indi->total_weights !=
            ptr_source_Dataset_Manager_received->total_weights ||
        indi->total_bias !=
            ptr_source_Dataset_Manager_received->total_bias) {
      if (indi->copy(*ptr_source_Dataset_Manager_received,
                                       true, true) == false) {
        ERR(L"An error has been triggered from the `copy(ptr, "
            L"true, true)` function.");

        return false;
      }
    } else if (indi->update(
                   *ptr_source_Dataset_Manager_received, true, true) == false) {
      ERR(L"An error has been triggered from the `update(ptr, "
          L"true, true)` function.");

      return false;
    }
  }

  return true;
}

bool Gaussian_Search::Update__Dataset_Manager(
    Datasets *const ptr_source_Dataset_Manager_received) {
  this->p_ptr_array_ptr_dataset_manager[0] =
      ptr_source_Dataset_Manager_received;

  Datasets *tmp_ptr_Dataset_Manager;

  for (size_t tmp_thread_index(1_UZ); tmp_thread_index != this->_number_threads;
       ++tmp_thread_index) {
    tmp_ptr_Dataset_Manager =
        this->p_ptr_array_ptr_dataset_manager[tmp_thread_index];

    if (tmp_ptr_Dataset_Manager->reference(
            ptr_source_Dataset_Manager_received) == false) {
      ERR(L"An error has been triggered from the "
          L"`reference(ptr)` function.");

      return false;
    }
  }

  return true;
}

bool Gaussian_Search::Enable__OpenMP__Population(void) {
  Model *indi;

  for (size_t k(0_UZ);
       k != this->_population_size; ++k) {
    indi =
        this->individuals[k];

    indi->use_mp = true;
  }

  return true;
}

bool Gaussian_Search::Disable__OpenMP__Population(void) {
  Model *indi;

  for (size_t k(0_UZ);
       k != this->_population_size; ++k) {
    indi =
        this->individuals[k];

    indi->use_mp = false;
  }

  return true;
}

bool Gaussian_Search::Optimize(size_t const number_iterations_received,
                               Datasets *const datasets, Model *const model) {
  if (this->_use_mp && this->_is_mp_initialized) {
    return (
        this->Optimize__OpenMP(number_iterations_received, datasets, model));
  } else {
    return (this->Optimize__Loop(number_iterations_received, datasets, model));
  }
}

bool Gaussian_Search::Optimize__Loop(size_t const number_iterations_received,
                                     Datasets *const datasets,
                                     Model *const model) {
  size_t tmp_iterations, k;

  if (this->Update__Population(model) == false) {
    ERR(L"An error has been triggered from the "
        L"`Update__Population(ptr)` function.");

    return false;
  } else if (this->Initialize__Hyper_Parameters(model) == false) {
    ERR(L"An error has been triggered from the "
        L"`Initialize__Hyper_Parameters(ptr)` function.");

    return false;
  } else if (this->Shuffle__Hyper_Parameter() == false) {
    ERR(L"An error has been triggered from the "
        L"`Shuffle__Hyper_Parameter()` function.");

    return false;
  } else if (this->Feed__Hyper_Parameter() == false) {
    ERR(L"An error has been triggered from the "
        L"`Feed__Hyper_Parameter()` function.");

    return false;
  }

  Model *indi;

  for (tmp_iterations = 0_UZ; tmp_iterations != number_iterations_received;
       ++tmp_iterations) {
    for (k = 0_UZ;
         k != this->_population_size;
         ++k) {
      indi =
          this->individuals[k];

#ifdef COMPILE_CUDA
      if (indi->Use__CUDA()) {
        datasets->Get__CUDA()->train(indi);
      } else
#endif
      {
        datasets->train(indi);
      }
    }
  }

  for (k = 0_UZ;
       k != this->_population_size; ++k) {
    indi =
        this->individuals[k];

#ifdef COMPILE_CUDA
    if (indi->Use__CUDA()) {
      datasets->Get__CUDA()->Type_Testing(ENV::TRAIN, indi);
    } else
#endif
    {
      datasets->Type_Testing(ENV::TRAIN, indi);
    }
  }

  return true;
}

bool Gaussian_Search::Optimize__OpenMP(size_t const number_iterations_received,
                                       Datasets *const datasets,
                                       Model *const model) {
  size_t tmp_iterations(0_UZ);

  int const tmp_population_size__int(static_cast<int>(this->_population_size));
  int tmp_individual_index__int(0);

  Model *indi(nullptr);

  Datasets *tmp_ptr_Dataset_Manager(nullptr);

  if (this->Update__Population(model) == false) {
    ERR(L"An error has been triggered from the "
        L"`Update__Population(ptr)` function.");

    return false;
  } else if (this->Update__Dataset_Manager(datasets) == false) {
    ERR(L"An error has been triggered from the "
        L"`Update__Dataset_Manager(ptr)` function.");

    return false;
  } else if (this->Initialize__Hyper_Parameters(model) == false) {
    ERR(L"An error has been triggered from the "
        L"`Initialize__Hyper_Parameters(ptr)` function.");

    return false;
  } else if (this->Shuffle__Hyper_Parameter() == false) {
    ERR(L"An error has been triggered from the "
        L"`Shuffle__Hyper_Parameter()` function.");

    return false;
  } else if (this->Feed__Hyper_Parameter() == false) {
    ERR(L"An error has been triggered from the "
        L"`Feed__Hyper_Parameter()` function.");

    return false;
  }

  // If the neural network use OpenMP, disable it.
  if (this->individuals[0]->is_mp_initialized &&
      this->Disable__OpenMP__Population() == false) {
    ERR(L"An error has been triggered from the "
        L"`Disable__OpenMP__Population()` function.");

    return false;
  }

  omp_set_num_threads(static_cast<int>(this->_number_threads));

#pragma omp parallel private(tmp_iterations, tmp_individual_index__int, \
                             indi, tmp_ptr_Dataset_Manager)
  {
    for (tmp_iterations = 0_UZ; tmp_iterations != number_iterations_received;
         ++tmp_iterations) {
#pragma omp for schedule(dynamic)
      for (tmp_individual_index__int = 0;
           tmp_individual_index__int < tmp_population_size__int;
           ++tmp_individual_index__int) {
        indi =
            this->individuals[tmp_individual_index__int];

        tmp_ptr_Dataset_Manager =
            this->p_ptr_array_ptr_dataset_manager[omp_get_thread_num()];

#ifdef COMPILE_CUDA
        if (indi->Use__CUDA()) {
          tmp_ptr_Dataset_Manager->Get__CUDA()->train(indi);
        } else
#endif
        {
          tmp_ptr_Dataset_Manager->train(indi);
        }
      }
    }

#pragma omp for schedule(dynamic)
    for (tmp_individual_index__int = 0;
         tmp_individual_index__int < tmp_population_size__int;
         ++tmp_individual_index__int) {
      indi =
          this->individuals[tmp_individual_index__int];

      tmp_ptr_Dataset_Manager =
          this->p_ptr_array_ptr_dataset_manager[omp_get_thread_num()];

#ifdef COMPILE_CUDA
      if (indi->Use__CUDA()) {
        tmp_ptr_Dataset_Manager->Get__CUDA()->Type_Testing(
            ENV::TRAIN, indi);
      } else
#endif
      {
        tmp_ptr_Dataset_Manager->Type_Testing(ENV::TRAIN,
                                              indi);
      }
    }
  }

  // If the neural network was using OpenMP, enable it.
  if (this->individuals[0]->is_mp_initialized &&
      this->Enable__OpenMP__Population() == false) {
    ERR(L"An error has been triggered from the "
        L"`Disable__OpenMP__Population()` function.");

    return false;
  }

  return true;
}

bool Gaussian_Search::Evaluation(void) {
  double loss(this->individuals[0]->get_loss(ENV::NONE));

  std::pair<int, double> best_model_info(0, loss);

  Model *indi;

  for (size_t k(1_UZ); k != this->_population_size; ++k) {
    indi = this->individuals[k];

    if (best_model_info.second >
        (loss = indi->get_loss(ENV::NONE))) {
      best_model_info.first = static_cast<int>(k);
      best_model_info.second = loss;
    }
  }

  if (best_model_info.first != 0 &&
      this->individuals[0]->update(
          *this->individuals[best_model_info.first], true,
          true) == false) {
    ERR(L"An error has been triggered from the "
        L"`update(ptr[%d], true, true)` function.",
        best_model_info.first);

    return false;
  }

  return true;
}

bool Gaussian_Search::Evaluation(Datasets *const datasets) {
  if (this->_use_mp && this->_is_mp_initialized) {
    return (this->Evaluation__OpenMP(datasets));
  } else {
    return (this->Evaluation__Loop(datasets));
  }
}

bool Gaussian_Search::Evaluation__Loop(Datasets *const datasets) {
  size_t k, k_best(0_UZ);

  Model *indi(this->individuals[0]);

#ifdef COMPILE_CUDA
  if (indi->Use__CUDA()) {
    datasets->Get__CUDA()->Type_Testing(
        datasets->Get__Type_Dataset_Evaluation(), indi);
  } else
#endif
  {
    datasets->Type_Testing(datasets->Get__Type_Dataset_Evaluation(),
                           indi);
  }

  for (k = 1_UZ;
       k != this->_population_size; ++k) {
    indi =
        this->individuals[k];

#ifdef COMPILE_CUDA
    if (indi->Use__CUDA()) {
      datasets->Get__CUDA()->Type_Testing(
          datasets->Get__Type_Dataset_Evaluation(), indi);
    } else
#endif
    {
      datasets->Type_Testing(datasets->Get__Type_Dataset_Evaluation(),
                             indi);
    }

    if (this->individuals[k_best]->Compare(
            datasets->Use__Metric_Loss(),
            datasets->Get__Dataset_In_Equal_Less_Holdout_Accepted(),
            datasets->Get__Type_Dataset_Evaluation(),
            datasets->Get__Minimum_Loss_Holdout_Accepted(),
            indi)) {
      k_best = k;
    }
  }

  if (k_best != 0_UZ &&
      this->individuals[0]->update(
          *this->individuals[k_best], true,
          true) == false) {
    ERR(L"An error has been triggered from the "
        L"`update(ptr[%zu], true, true)` function.",
        k_best);

    return false;
  }

  return true;
}

bool Gaussian_Search::Evaluation__OpenMP(Datasets *const datasets) {
  int const tmp_population_size__int(static_cast<int>(this->_population_size));
  int tmp_individual_index__int(0), tmp_best_individual_index__int(0);

  Model *indi(nullptr);

  Datasets *tmp_ptr_Dataset_Manager(nullptr);

  if (this->Update__Dataset_Manager(datasets) == false) {
    ERR(L"An error has been triggered from the "
        L"`Update__Dataset_Manager(ptr)` function.");

    return false;
  }

  // If the neural network use OpenMP, disable it.
  if (this->individuals[0]->is_mp_initialized &&
      this->Disable__OpenMP__Population() == false) {
    ERR(L"An error has been triggered from the "
        L"`Disable__OpenMP__Population()` function.");

    return false;
  }

  omp_set_num_threads(static_cast<int>(this->_number_threads));

#pragma omp parallel private(indi, tmp_ptr_Dataset_Manager)
  {
#pragma omp single nowait
    {
      indi = this->individuals[0];

      tmp_ptr_Dataset_Manager =
          this->p_ptr_array_ptr_dataset_manager[omp_get_thread_num()];

#ifdef COMPILE_CUDA
      if (indi->Use__CUDA()) {
        tmp_ptr_Dataset_Manager->Get__CUDA()->Type_Testing(
            tmp_ptr_Dataset_Manager->Get__Type_Dataset_Evaluation(),
            indi);
      } else
#endif
      {
        tmp_ptr_Dataset_Manager->Type_Testing(
            tmp_ptr_Dataset_Manager->Get__Type_Dataset_Evaluation(),
            indi);
      }
    }

#pragma omp for schedule(dynamic)
    for (tmp_individual_index__int = 1;
         tmp_individual_index__int < tmp_population_size__int;
         ++tmp_individual_index__int) {
      indi =
          this->individuals[tmp_individual_index__int];

      tmp_ptr_Dataset_Manager =
          this->p_ptr_array_ptr_dataset_manager[omp_get_thread_num()];

#ifdef COMPILE_CUDA
      if (indi->Use__CUDA()) {
        tmp_ptr_Dataset_Manager->Get__CUDA()->Type_Testing(
            tmp_ptr_Dataset_Manager->Get__Type_Dataset_Evaluation(),
            indi);
      } else
#endif
      {
        tmp_ptr_Dataset_Manager->Type_Testing(
            tmp_ptr_Dataset_Manager->Get__Type_Dataset_Evaluation(),
            indi);
      }

#pragma omp critical
      if (this->individuals[tmp_best_individual_index__int]
              ->Compare(
                  tmp_ptr_Dataset_Manager->Use__Metric_Loss(),
                  tmp_ptr_Dataset_Manager
                      ->Get__Dataset_In_Equal_Less_Holdout_Accepted(),
                  tmp_ptr_Dataset_Manager->Get__Type_Dataset_Evaluation(),
                  tmp_ptr_Dataset_Manager->Get__Minimum_Loss_Holdout_Accepted(),
                  indi)) {
        tmp_best_individual_index__int = tmp_individual_index__int;
      }
    }
  }

  // If the neural network was using OpenMP, enable it.
  if (this->individuals[0]->is_mp_initialized &&
      this->Enable__OpenMP__Population() == false) {
    ERR(L"An error has been triggered from the "
        L"`Disable__OpenMP__Population()` function.");

    return false;
  }

  if (tmp_best_individual_index__int != 0 &&
      this->individuals[0]->update(
          *this->individuals[tmp_best_individual_index__int],
          true, true) == false) {
    ERR(L"An error has been triggered from the `update(ptr[%d], "
        L"true, true)` function.",
        tmp_best_individual_index__int);

    return false;
  }

  return true;
}

bool Gaussian_Search::user_controls(void) {
  while (true) {
    INFO(L"");
    INFO(L"User controls:");
    INFO(L"[0]: Population size (%zu).", this->_population_size);
    INFO(L"[1]: Population gaussian (%f).", this->_population_gaussian_percent);
    INFO(L"[2]: Add Hyperparameter.");
    INFO(L"[3]: Modify Hyperparameter.");
    INFO(L"[4]: OpenMP.");
    INFO(L"[5]: Quit.");

    switch (parse_discrete(0, 5, L"Option: ")) {
      case 0:
        INFO(L"");
        INFO(L"Population size:");
        INFO(L"Range[1, 8].");
        INFO(L"default=60.");
        if (this->Set__Population_Size(
                parse_discrete(0_UZ, L"Population size: ")) == false) {
          ERR(L"An error has been triggered from the "
              L"`Set__Population_Size()` function.");

          return false;
        }
        break;
      case 1:
        INFO(L"");
        INFO(L"Population gaussian in percent:");
        INFO(L"Range[2.0, 100.0].");
        INFO(L"default=75.0%%.");
        if (this->Set__Population_Gaussian(parse_real<double>(
                2.0, 100.0, L"Population gaussian (percent): ")) == false) {
          ERR(L"An error has been triggered from the "
              L"`Set__Population_Gaussian()` function.");

          return false;
        }
        break;
      case 2:
        if (this->User_Controls__Push_Back() == false) {
          ERR(L"An error has been triggered from the "
              L"`User_Controls__Push_Back()` function.");

          return false;
        }
        break;
      case 3:
        if (this->User_Controls__Hyperparameter_Manager() == false) {
          ERR(L"An error has been triggered from the "
              L"`User_Controls__Hyperparameter()` function.");

          return false;
        }
        break;
      case 4:
        if (this->User_Controls__OpenMP() == false) {
          ERR(L"An error has been triggered from the "
              L"`User_Controls__OpenMP()` function.");

          return false;
        }
        break;
      case 5:
        return true;
      default:
        ERR(L"An error has been triggered from the "
            L"`parse_discrete(%d, %d)` function.",
            0, 5);
        return false;
    }
  }

  return false;
}

bool Gaussian_Search::User_Controls__Push_Back(void) {
  int tmp_option;

  size_t tmp_layer_index;

  real minval, maxval, variance;

  while (true) {
    INFO(L"");
    INFO(L"User controls, Hyper parameter push back:");
    INFO(L"[0]: Regularization, weight decay.");
    INFO(L"[1]: Regularization, L1.");
    INFO(L"[2]: Regularization, L2.");
    INFO(L"[3]: Regularization, max-norm constraints.");
    INFO(L"[4]: Normalization, average momentum.");
    INFO(L"[5]: Dropout, alpha, dropout probability.");
    INFO(L"[6]: Dropout, alpha, a.");
    INFO(L"[7]: Dropout, alpha, b.");
    INFO(L"[8]: Dropout, bernoulli, keep probability.");
    INFO(L"[9]: Dropout, bernoulli-inverted, keep probability.");
    INFO(L"[10]: Dropout, gaussian, dropout probability.");
    INFO(L"[11]: Dropout, uout, dropout probability.");
    INFO(L"[12]: Dropout, zoneout, cell zoneout probability.");
    INFO(L"[13]: Dropout, zoneout, hidden zoneout probability.");
    INFO(L"[14]: Quit.");

    if ((tmp_option = parse_discrete(0, 14, L"Option: ")) == 14) {
      return true;
    }

    INFO(L"");
    INFO(L"Variance.");
    INFO(L"Range[1e-7, 8].");
    variance = parse_real(1e-7_r, L"Variance: ");

    switch (tmp_option) {
      case 0:  // Regularization, Weight decay.
      case 1:  // Regularization, L1.
      case 2:  // Regularization, L2.
        INFO(L"");
        INFO(L"Minimum value.");
        INFO(L"Range[0, 1].");
        INFO(L"default=0.");
        minval = parse_real(0_r, 1_r, L"Minimum value: ");

        INFO(L"");
        INFO(L"Maximum value.");
        INFO(L"Range[%f, 1].", minval);
        INFO(L"default=1.");
        maxval =
            parse_real(minval, 1_r, L"Maximum value: ");

        if (this->push_back(
                tmp_option, 0_UZ, -(std::numeric_limits<real>::max)(),
                minval, maxval, variance) == false) {
          ERR(L"An error has been triggered from the "
              L"`push_back()` function.");

          return false;
        }
        break;
      case 3:  // Regularization, Max-norm constraints.
        INFO(L"");
        INFO(L"Minimum value.");
        INFO(L"Range[0, 8].");
        INFO(L"default=0.");
        minval = parse_real(0_r, L"Minimum value: ");

        INFO(L"");
        INFO(L"Maximum value.");
        INFO(L"Range[%f, 8].", minval);
        INFO(L"default=16.");
        maxval =
            parse_real(minval, L"Maximum value: ");

        if (this->push_back(3, 0_UZ, -(std::numeric_limits<real>::max)(),
                            minval, maxval,
                            variance) == false) {
          ERR(L"An error has been triggered from the "
              L"`push_back()` function.");

          return false;
        }
        break;
      case 4:  // Normalization, average momentum.
        INFO(L"");
        INFO(L"Minimum value.");
        INFO(L"Range[0, %f].", cast(1_r - 1e-7_r));
        INFO(L"default=0.");
        minval =
            parse_real(0_r, 1_r - 1e-7_r, L"Minimum value: ");

        INFO(L"");
        INFO(L"Maximum value.");
        INFO(L"Range[%f, %f].", minval, cast(1_r - 1e-7_r));
        INFO(L"default=0.999.");
        maxval = parse_real(
            minval, 1_r - 1e-7_r, L"Maximum value: ");

        if (this->push_back(4, 0_UZ, -(std::numeric_limits<real>::max)(),
                            minval, maxval,
                            variance) == false) {
          ERR(L"An error has been triggered from the "
              L"`push_back()` function.");

          return false;
        }
        break;
      case 5:   // Dropout, alpha, dropout probability.
      case 6:   // Dropout, alpha, a.
      case 7:   // Dropout, alpha, b.
      case 8:   // Dropout, bernoulli, keep probability.
      case 9:   // Dropout, bernoulli-inverted, keep probability.
      case 10:  // Dropout, gaussian, dropout probability.
      case 11:  // Dropout, uout, dropout probability.
      case 12:  // Dropout, zoneout, cell zoneout probability.
      case 13:  // Dropout, zoneout, hidden zoneout probability.
        INFO(L"");
        tmp_layer_index = parse_discrete(0_UZ, L"Layer index: ");

        INFO(L"");
        INFO(L"Minimum value.");
        INFO(L"Range[0, 1].");
        INFO(L"default=0.");
        minval = parse_real(0_r, 1_r, L"Minimum value: ");

        INFO(L"");
        INFO(L"Maximum value.");
        INFO(L"Range[%f, 1].", minval);
        INFO(L"default=1.");
        maxval =
            parse_real(minval, 1_r, L"Maximum value: ");

        if (this->push_back(tmp_option, tmp_layer_index,
                            -(std::numeric_limits<real>::max)(),
                            minval, maxval,
                            variance) == false) {
          ERR(L"An error has been triggered from the "
              L"`push_back()` function.");

          return false;
        }
        break;
      default:
        ERR(L"An error has been triggered from the "
            "`parse_discrete(%d, %d)` function.",
            0, 14);
        return false;
    }
  }

  return false;
}

std::wstring Gaussian_Search::Get__ID_To_String(
    int const hyperparameter_id_received) const {
  switch (hyperparameter_id_received) {
    case 0:
      return L"Regularization, weight decay.";
    case 1:
      return L"Regularization, L1.";
    case 2:
      return L"Regularization, L2.";
    case 3:
      return L"Regularization, max-norm constraints.";
    case 4:
      return L"Normalization, average momentum.";
    case 5:
      return L"Dropout, alpha, dropout probability.";
    case 6:
      return L"Dropout, alpha, a.";
    case 7:
      return L"Dropout, alpha, b.";
    case 8:
      return L"Dropout, bernoulli, keep probability.";
    case 9:
      return L"Dropout, bernoulli-inverted, keep probability.";
    case 10:
      return L"Dropout, gaussian, dropout probability.";
    case 11:
      return L"Dropout, uout, dropout probability.";
    case 12:
      return L"Dropout, zoneout, cell zoneout probability.";
    case 13:
      return L"Dropout, zoneout, hidden zoneout probability.";
    default:
      ERR(L"Hyper parameter id (%d) is not managed in the "
          L"switch.",
          hyperparameter_id_received);
      return L"";
  }
}

bool Gaussian_Search::User_Controls__Hyperparameter_Manager(void) {
  size_t tmp_option, tmp_layer_index;

  real minval, maxval, variance;

  while (true) {
    INFO(L"");
    INFO(L"User controls, hyperparameter manager.");
    for (size_t tmp_hyperparameter_index(0_UZ);
         tmp_hyperparameter_index != this->_vector_hyperparameters.size();
         ++tmp_hyperparameter_index) {
      INFO(L"[%zu]: %ls (%d, %zu, %f, %f, %f, %f).", tmp_hyperparameter_index,
           this
               ->Get__ID_To_String(std::get<0>(
                   this->_vector_hyperparameters[tmp_hyperparameter_index]))
               .c_str(),
           std::get<0>(this->_vector_hyperparameters[tmp_hyperparameter_index]),
           std::get<1>(this->_vector_hyperparameters[tmp_hyperparameter_index]),
           cast(std::get<2>(
               this->_vector_hyperparameters[tmp_hyperparameter_index])),
           cast(std::get<3>(
               this->_vector_hyperparameters[tmp_hyperparameter_index])),
           cast(std::get<4>(
               this->_vector_hyperparameters[tmp_hyperparameter_index])),
           cast(std::get<5>(
               this->_vector_hyperparameters[tmp_hyperparameter_index])));
    }

    INFO(L"[%zu]: Quit.", this->_vector_hyperparameters.size());

    if ((tmp_option = parse_discrete(
             0_UZ, this->_vector_hyperparameters.size(), L"Option: ")) <=
        this->_vector_hyperparameters.size()) {
      if (tmp_option == this->_vector_hyperparameters.size()) {
        return true;
      }

      tmp_layer_index = 0_UZ;

      minval = 0_r;
      maxval = 1_r;

      INFO(L"");
      INFO(L"Variance.");
      INFO(L"Range[1e-7, 8].");
      variance = parse_real(1e-7_r, L"Variance: ");

      switch (std::get<0>(this->_vector_hyperparameters[tmp_option])) {
        case 0:  // Regularization, Weight decay.
        case 1:  // Regularization, L1.
        case 2:  // Regularization, L2.
          INFO(L"");
          INFO(L"Minimum value.");
          INFO(L"Range[0, 1].");
          INFO(L"default=0.");
          minval =
              parse_real(0_r, 1_r, L"Minimum value: ");

          INFO(L"");
          INFO(L"Maximum value.");
          INFO(L"Range[%f, 1].", minval);
          INFO(L"default=1.");
          maxval =
              parse_real(minval, 1_r, L"Maximum value: ");
          break;
        case 3:  // Regularization, Max-norm constraints.
          INFO(L"");
          INFO(L"Minimum value.");
          INFO(L"Range[0, 8].");
          INFO(L"default=0.");
          minval = parse_real(0_r, L"Minimum value: ");

          INFO(L"");
          INFO(L"Maximum value.");
          INFO(L"Range[%f, 8].", minval);
          INFO(L"default=16.");
          maxval =
              parse_real(minval, L"Maximum value: ");
          break;
        case 4:  // Normalization, average momentum.
          INFO(L"");
          INFO(L"Minimum value.");
          INFO(L"Range[0, %f].", cast(1_r - 1e-7_r));
          INFO(L"default=0.");
          minval =
              parse_real(0_r, 1_r - 1e-7_r, L"Minimum value: ");

          INFO(L"");
          INFO(L"Maximum value.");
          INFO(L"Range[%f, %f].", minval,
               cast(1_r - 1e-7_r));
          INFO(L"default=0.999.");
          maxval = parse_real(
              minval, 1_r - 1e-7_r, L"Maximum value: ");
          break;
        case 5:   // Dropout, alpha, dropout probability.
        case 6:   // Dropout, alpha, a.
        case 7:   // Dropout, alpha, b.
        case 8:   // Dropout, bernoulli, keep probability.
        case 9:   // Dropout, bernoulli-inverted, keep probability.
        case 10:  // Dropout, gaussian, dropout probability.
        case 11:  // Dropout, uout, dropout probability.
        case 12:  // Dropout, zoneout, cell zoneout probability.
        case 13:  // Dropout, zoneout, hidden zoneout probability.
          INFO(L"");
          tmp_layer_index = parse_discrete(0_UZ, L"Layer index: ");

          INFO(L"");
          INFO(L"Minimum value.");
          INFO(L"Range[0, 1].");
          INFO(L"default=0.");
          minval =
              parse_real(0_r, 1_r, L"Minimum value: ");

          INFO(L"");
          INFO(L"Maximum value.");
          INFO(L"Range[%f, 1].", minval);
          INFO(L"default=1.");
          maxval =
              parse_real(minval, 1_r, L"Maximum value: ");
          break;
        default:
          ERR(L"An error has been triggered from the "
              L"`parse_discrete(%d, %d)` function.",
              0, 14);
          return false;
      }

      std::get<1>(this->_vector_hyperparameters[tmp_option]) = tmp_layer_index;
      std::get<3>(this->_vector_hyperparameters[tmp_option]) =
          minval;
      std::get<4>(this->_vector_hyperparameters[tmp_option]) =
          maxval;
      std::get<5>(this->_vector_hyperparameters[tmp_option]) = variance;
    } else {
      ERR(L"An error has been triggered from the "
          L"`parse_discrete(%zu, %zu)` function.",
          0_UZ, this->_vector_hyperparameters.size());
    }
  }

  return false;
}

bool Gaussian_Search::User_Controls__OpenMP(void) {
  while (true) {
    INFO(L"");
    INFO(L"User controls, OpenMP:");
    INFO(L"[0]: Use OpenMP (%ls | %ls).", to_wstring(this->_use_mp).c_str(),
         to_wstring(this->_is_mp_initialized).c_str());
    INFO(L"[1]: Maximum threads (%.2f%%).",
         this->_percentage_maximum_thread_usage);
    INFO(L"[2]: Quit.");

    switch (parse_discrete(0, 2, L"Option: ")) {
      case 0:
        INFO(L"");
        if (this->set_mp(accept(L"Use OpenMP: ")) == false) {
          ERR(L"An error has been triggered from the "
              L"`set_mp()` function.");

          return false;
        }
        break;
      case 1:
        INFO(L"");
        INFO(L"Maximum threads:");
        INFO(L"Range[0.0%%, 100.0%%].");
        if (this->Set__Maximum_Thread_Usage(parse_real(
                0_r, 100_r, L"Maximum threads (percent): ")) == false) {
          ERR(L"An error has been triggered from the "
              L"`Set__Maximum_Thread_Usage()` function.");

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

  return false;
}

bool Gaussian_Search::push_back(int const para_id,
                                size_t const index,
                                real const value_received,
                                real const minval,
                                real const maxval,
                                real const variance) {
  if (para_id >= 14) {
    ERR(L"Hyperparameter id (%d) undefined.", para_id);

    return false;
  } else if (minval >= maxval) {
    ERR(L"Minimum value (%f) can not be greater or equal to "
        L"maximum value (%f).",
        minval, maxval);

    return false;
  } else if (variance <= 0_r) {
    ERR(L"Variance can not be less or equal to zero.");

    return false;
  } else if (variance >
             maxval - minval) {
    ERR(L"Variance (%f) can not be greater to than %f.",
        variance, maxval - minval);

    return false;
  }

  this->_vector_hyperparameters.push_back(std::tuple<int, size_t, real, real, real, real>(
      para_id, index, value_received,
      minval, maxval, variance));

  return true;
}

bool Gaussian_Search::Initialize__Hyper_Parameters(Model *const model) {
  for (size_t tmp_hyper_parameter_index(0_UZ);
       tmp_hyper_parameter_index != this->_vector_hyperparameters.size();
       ++tmp_hyper_parameter_index) {
    if (this->Initialize__Hyper_Parameter(
            this->_vector_hyperparameters[tmp_hyper_parameter_index], model) ==
        false) {
      ERR(L"An error has been triggered from the "
          L"`Initialize__Hyper_Parameter(hyper[%zu], ptr)` function.",
          tmp_hyper_parameter_index);

      return false;
    }
  }

  return true;
}

bool Gaussian_Search::Initialize__Hyper_Parameter(
    std::tuple<int, size_t, real, real, real, real> &ref_hyperparameter_tuple_received,
    Model *const model) {
  struct Layer const *layer_it;

  switch (std::get<0>(ref_hyperparameter_tuple_received)) {
    case 0:  // Regularization, Weight decay.
      std::get<2>(ref_hyperparameter_tuple_received) = model->weight_decay;
      break;
    case 1:  // Regularization, L1.
      std::get<2>(ref_hyperparameter_tuple_received) =
          model->regularization__l1;
      break;
    case 2:  // Regularization, L2.
      std::get<2>(ref_hyperparameter_tuple_received) =
          model->regularization__l2;
      break;
    case 3:  // Regularization, Max-norm constraints.
      std::get<2>(ref_hyperparameter_tuple_received) =
          model->regularization__max_norm_constraints;
      break;
    case 4:  // Normalization, average momentum.
      std::get<2>(ref_hyperparameter_tuple_received) =
          model->normalization_momentum_average;
      break;
    case 5:   // Dropout, alpha, dropout probability.
    case 8:   // Dropout, bernoulli, keep probability.
    case 9:   // Dropout, bernoulli-inverted, keep probability.
    case 10:  // Dropout, gaussian, dropout probability.
    case 11:  // Dropout, uout, dropout probability.
    case 12:  // Dropout, zoneout, cell zoneout probability.
      layer_it = model->ptr_array_layers +
                         std::get<1>(ref_hyperparameter_tuple_received);

      std::get<2>(ref_hyperparameter_tuple_received) =
          layer_it->dropout_values[0];
      break;
    case 6:   // Dropout, alpha, a.
    case 13:  // Dropout, zoneout, hidden zoneout probability.
      layer_it = model->ptr_array_layers +
                         std::get<1>(ref_hyperparameter_tuple_received);

      std::get<2>(ref_hyperparameter_tuple_received) =
          layer_it->dropout_values[1];
      break;
    case 7:  // Dropout, alpha, b.
      layer_it = model->ptr_array_layers +
                         std::get<1>(ref_hyperparameter_tuple_received);

      std::get<2>(ref_hyperparameter_tuple_received) =
          layer_it->dropout_values[2];
      break;
    default:
      ERR(L"Hyper parameter id (%d) is not managed in the "
          L"switch.",
          std::get<0>(ref_hyperparameter_tuple_received));
      return false;
  }

  // If value underflow.
  std::get<2>(ref_hyperparameter_tuple_received) =
      std::max(std::get<2>(ref_hyperparameter_tuple_received),
               std::get<3>(ref_hyperparameter_tuple_received));

  // If value overflow.
  std::get<2>(ref_hyperparameter_tuple_received) =
      std::min(std::get<2>(ref_hyperparameter_tuple_received),
               std::get<4>(ref_hyperparameter_tuple_received));

  return true;
}

bool Gaussian_Search::Shuffle__Hyper_Parameter(void) {
  if (this->_vector_hyperparameters.size() == 0_UZ) {
    ERR(L"No hyper parameter available for shuffling.");
    return false;
  }

  this->int_gen.range(
      0, static_cast<int>(this->_vector_hyperparameters.size()) - 1);

  this->_ptr_selected_hyperparameter =
      &this->_vector_hyperparameters.at(this->int_gen());

  return true;
}

bool Gaussian_Search::Feed__Hyper_Parameter(void) {
  size_t const tmp_population_random_size(std::max<size_t>(
      1_UZ, static_cast<size_t>(
                floor(static_cast<double>(this->_population_size) *
                      (100.0 - this->_population_gaussian_percent) / 100.0))));
  size_t k;

  real const default_val(std::get<2>(*this->_ptr_selected_hyperparameter)),
      minval(std::get<3>(*this->_ptr_selected_hyperparameter)),
      maxval(std::get<4>(*this->_ptr_selected_hyperparameter)),
      variance(std::get<5>(*this->_ptr_selected_hyperparameter));
  real val;

  std::tuple<int, size_t, real, real, real, real> tmp_hyperparameter_tuple(
      *this->_ptr_selected_hyperparameter);

  Model *indi;

  // Initialize random generator.
  this->real_gen.range(minval, maxval);

  // Exploration.
  for (k = 1_UZ;
       k != tmp_population_random_size;
       ++k) {
    std::get<2>(tmp_hyperparameter_tuple) =
        this->real_gen();

    indi =
        this->individuals[k];

    if (this->Feed__Hyper_Parameter(tmp_hyperparameter_tuple,
                                    indi) == false) {
      ERR(L"An error has been triggered from the "
          L"`Feed__Hyper_Parameter(ref, ptr)` function.");

      return false;
    }
  }

  // Initialize gaussian generator.
  this->gaussian.range(0_r, variance);

  // Exploitation.
  for (; k != this->_population_size;
       ++k) {
    do {
      val =
          default_val + this->gaussian();
    } while (val < minval ||
             val > maxval);

    std::get<2>(tmp_hyperparameter_tuple) = val;

    indi =
        this->individuals[k];

    if (this->Feed__Hyper_Parameter(tmp_hyperparameter_tuple,
                                    indi) == false) {
      ERR(L"An error has been triggered from the "
          L"`Feed__Hyper_Parameter(ref, ptr)` function.");

      return false;
    }
  }

  return true;
}

bool Gaussian_Search::Feed__Hyper_Parameter(
    std::tuple<int, size_t, real, real, real, real> const
        &ref_hyperparameter_tuple_received,
    Model *const model) {
  struct Layer const *layer_it;

  switch (std::get<0>(ref_hyperparameter_tuple_received)) {
    case 0:  // Regularization, weight decay.
      if (model->set_weight_decay(
              std::get<2>(ref_hyperparameter_tuple_received)) == false) {
        ERR(L"An error has been triggered from the "
            L"`set_weight_decay(%f)` function.",
            std::get<2>(ref_hyperparameter_tuple_received));

        return false;
      }
      break;
    case 1:  // Regularization, L1.
      if (model->set_l1(
              std::get<2>(ref_hyperparameter_tuple_received)) == false) {
        ERR(L"An error has been triggered from the "
            L"`set_l1(%f)` function.",
            std::get<2>(ref_hyperparameter_tuple_received));

        return false;
      }
      break;
    case 2:  // Regularization, L2.
      if (model->set_l2(
              std::get<2>(ref_hyperparameter_tuple_received)) == false) {
        ERR(L"An error has been triggered from the "
            L"`set_l2(%f)` function.",
            std::get<2>(ref_hyperparameter_tuple_received));

        return false;
      }
      break;
    case 3:  // Regularization, max-norm constraints.
      if (model->Set__Regularization__Max_Norm_Constraints(
              std::get<2>(ref_hyperparameter_tuple_received)) == false) {
        ERR(L"An error has been triggered from the "
            L"`Set__Regularization__Max_Norm_Constraints(%f)` function.",
            std::get<2>(ref_hyperparameter_tuple_received));

        return false;
      }
      break;
    case 4:  // Normalization, average momentum.
      if (model->Set__Normalization_Momentum_Average(
              std::get<2>(ref_hyperparameter_tuple_received)) == false) {
        ERR(L"An error has been triggered from the "
            L"`Set__Normalization_Momentum_Average(%f)` function.",
            std::get<2>(ref_hyperparameter_tuple_received));

        return false;
      }
      break;
    case 5:  // Dropout, alpha, dropout probability.
      layer_it = model->ptr_array_layers +
                         std::get<1>(ref_hyperparameter_tuple_received);

      if (model->set_dropout(
              std::get<1>(ref_hyperparameter_tuple_received),
              LAYER_DROPOUT::ALPHA,
              std::array<real, 3_UZ>{
                  std::get<2>(ref_hyperparameter_tuple_received),
                  layer_it->dropout_values[1],
                  layer_it->dropout_values[2]}
                  .data()) == false) {
        ERR(L"An error has been triggered from the "
            L"`set_dropout(%zu, %u, %f, %f, %f)` function.",
            std::get<1>(ref_hyperparameter_tuple_received),
            LAYER_DROPOUT::ALPHA,
            std::get<2>(ref_hyperparameter_tuple_received),
            layer_it->dropout_values[1],
            layer_it->dropout_values[2]);

        return false;
      }
      break;
    case 6:  // Dropout, alpha, a.
      layer_it = model->ptr_array_layers +
                         std::get<1>(ref_hyperparameter_tuple_received);

      if (model->set_dropout(
              std::get<1>(ref_hyperparameter_tuple_received),
              LAYER_DROPOUT::ALPHA,
              std::array<real, 3_UZ>{
                  layer_it->dropout_values[0],
                  std::get<2>(ref_hyperparameter_tuple_received),
                  layer_it->dropout_values[2]}
                  .data()) == false) {
        ERR(L"An error has been triggered from the "
            L"`set_dropout(%zu, %u, %f, %f, %f)` function.",
            std::get<1>(ref_hyperparameter_tuple_received),
            LAYER_DROPOUT::ALPHA, layer_it->dropout_values[0],
            std::get<2>(ref_hyperparameter_tuple_received),
            layer_it->dropout_values[2]);

        return false;
      }
      break;
    case 7:  // Dropout, alpha, b.
      layer_it = model->ptr_array_layers +
                         std::get<1>(ref_hyperparameter_tuple_received);

      if (model->set_dropout(
              std::get<1>(ref_hyperparameter_tuple_received),
              LAYER_DROPOUT::ALPHA,
              std::array<real, 3_UZ>{
                  layer_it->dropout_values[0],
                  layer_it->dropout_values[1],
                  std::get<2>(ref_hyperparameter_tuple_received)}
                  .data()) == false) {
        ERR(L"An error has been triggered from the "
            L"`set_dropout(%zu, %u, %f, %f, %f)` function.",
            std::get<1>(ref_hyperparameter_tuple_received),
            LAYER_DROPOUT::ALPHA, layer_it->dropout_values[0],
            layer_it->dropout_values[1],
            std::get<2>(ref_hyperparameter_tuple_received));

        return false;
      }
      break;
    case 8:  // Dropout, bernoulli, keep probability.
      if (model->set_dropout(
              std::get<1>(ref_hyperparameter_tuple_received),
              LAYER_DROPOUT::BERNOULLI,
              std::array<real, 1_UZ>{
                  std::get<2>(ref_hyperparameter_tuple_received)}
                  .data()) == false) {
        ERR(L"An error has been triggered from the "
            L"`set_dropout(%zu, %u, %f)` function.",
            std::get<1>(ref_hyperparameter_tuple_received),
            LAYER_DROPOUT::BERNOULLI,
            std::get<2>(ref_hyperparameter_tuple_received));

        return false;
      }
      break;
    case 9:  // Dropout, bernoulli-inverted, keep probability.
      if (model->set_dropout(
              std::get<1>(ref_hyperparameter_tuple_received),
              LAYER_DROPOUT::BERNOULLI_INVERTED,
              std::array<real, 1_UZ>{
                  std::get<2>(ref_hyperparameter_tuple_received)}
                  .data()) == false) {
        ERR(L"An error has been triggered from the "
            L"`set_dropout(%zu, %u, %f)` function.",
            std::get<1>(ref_hyperparameter_tuple_received),
            LAYER_DROPOUT::BERNOULLI_INVERTED,
            std::get<2>(ref_hyperparameter_tuple_received));

        return false;
      }
      break;
    case 10:  // Dropout, gaussian, dropout probability.
      if (model->set_dropout(
              std::get<1>(ref_hyperparameter_tuple_received),
              LAYER_DROPOUT::GAUSSIAN,
              std::array<real, 1_UZ>{
                  std::get<2>(ref_hyperparameter_tuple_received)}
                  .data()) == false) {
        ERR(L"An error has been triggered from the "
            L"`set_dropout(%zu, %u, %f)` function.",
            std::get<1>(ref_hyperparameter_tuple_received),
            LAYER_DROPOUT::GAUSSIAN,
            std::get<2>(ref_hyperparameter_tuple_received));

        return false;
      }
      break;
    case 11:  // Dropout, uout, dropout probability.
      if (model->set_dropout(
              std::get<1>(ref_hyperparameter_tuple_received),
              LAYER_DROPOUT::UOUT,
              std::array<real, 1_UZ>{
                  std::get<2>(ref_hyperparameter_tuple_received)}
                  .data()) == false) {
        ERR(L"An error has been triggered from the "
            L"`set_dropout(%zu, %u, %f)` function.",
            std::get<1>(ref_hyperparameter_tuple_received), LAYER_DROPOUT::UOUT,
            std::get<2>(ref_hyperparameter_tuple_received));

        return false;
      }
      break;
    case 12:  // Dropout, zoneout, cell zoneout probability.
      layer_it = model->ptr_array_layers +
                         std::get<1>(ref_hyperparameter_tuple_received);

      if (model->set_dropout(
              std::get<1>(ref_hyperparameter_tuple_received),
              LAYER_DROPOUT::ZONEOUT,
              std::array<real, 2_UZ>{
                  std::get<2>(ref_hyperparameter_tuple_received),
                  layer_it->dropout_values[1]}
                  .data()) == false) {
        ERR(L"An error has been triggered from the "
            L"`set_dropout(%zu, %u, %f, %f)` function.",
            std::get<1>(ref_hyperparameter_tuple_received),
            LAYER_DROPOUT::ZONEOUT,
            std::get<2>(ref_hyperparameter_tuple_received),
            layer_it->dropout_values[1]);

        return false;
      }
      break;
    case 13:  // Dropout, zoneout, hidden zoneout probability.
      layer_it = model->ptr_array_layers +
                         std::get<1>(ref_hyperparameter_tuple_received);

      if (model->set_dropout(
              std::get<1>(ref_hyperparameter_tuple_received),
              LAYER_DROPOUT::ZONEOUT,
              std::array<real, 2_UZ>{
                  layer_it->dropout_values[0],
                  std::get<2>(ref_hyperparameter_tuple_received)}
                  .data()) == false) {
        ERR(L"An error has been triggered from the "
            L"`set_dropout(%zu, %u, %f, %f)` function.",
            std::get<1>(ref_hyperparameter_tuple_received),
            LAYER_DROPOUT::ZONEOUT, layer_it->dropout_values[0],
            std::get<2>(ref_hyperparameter_tuple_received));

        return false;
      }
      break;
    default:
      ERR(L"Hyper parameter id (%d) is not managed in the "
          L"switch.",
          std::get<0>(ref_hyperparameter_tuple_received));
      return false;
  }

  return true;
}

bool Gaussian_Search::Deinitialize__OpenMP(void) {
  if (this->_is_mp_initialized) {
    this->Deallocate__Dataset_Manager();

    this->_cache_number_threads = this->_number_threads = 0_UZ;

    this->_is_mp_initialized = false;
  }

  return true;
}

void Gaussian_Search::Deallocate__Dataset_Manager(void) {
  SAFE_DELETE_ARRAY(this->p_ptr_array_dataset_manager);
  SAFE_DELETE_ARRAY(this->p_ptr_array_ptr_dataset_manager);
}

void Gaussian_Search::Deallocate__Population(void) {
  SAFE_DELETE_ARRAY(this->p_ptr_array_individuals);
  SAFE_DELETE_ARRAY(this->individuals);
}

void Gaussian_Search::Deallocate(void) {
  this->Deallocate__Dataset_Manager();
  this->Deallocate__Population();
}

Gaussian_Search::~Gaussian_Search(void) { this->Deallocate(); }
}  // namespace DL