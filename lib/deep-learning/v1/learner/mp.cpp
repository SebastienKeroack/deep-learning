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

#include "deep-learning/v1/learner/model.hpp"
#include "deep-learning/data/string.hpp"
#include "deep-learning/data/time.hpp"
#include "deep-learning/io/logger.hpp"

#include <omp.h>

namespace DL::v1 {
void Model::Initialize__OpenMP(void)
{
    if(this->is_mp_initialized == false)
    {
        this->is_mp_initialized = true;

        omp_set_dynamic(0);
    }
}

bool Model::set_mp(bool const use_openmp_received)
{
    if((this->use_mp == false && use_openmp_received)
      ||
      (this->use_mp && use_openmp_received && this->is_mp_initialized == false))
    { this->Initialize__OpenMP(); }
    else if((this->use_mp && use_openmp_received == false)
              ||
              (this->use_mp == false && use_openmp_received == false && this->is_mp_initialized))
    {
        if(this->Deinitialize__OpenMP() == false) {
          ERR(L"An error has been triggered from the "
              L"`Deinitialize__OpenMP()` function.");
            return false;
        }
    }

    this->use_mp = use_openmp_received;

    return true;
}

bool Model::Set__Maximum_Thread_Usage(double const percentage_maximum_thread_usage_received)
{
    if(this->pct_threads == percentage_maximum_thread_usage_received) { return true; }

    this->pct_threads = percentage_maximum_thread_usage_received;

    if(this->update_mem_thread_size(this->cache_number_threads) == false) {
        ERR(L"An error has been triggered from the "
            L"`update_mem_thread_size(%zu)` function.",
            this->cache_number_threads);

        return false;
    }

    return true;
}

bool Model::Deinitialize__OpenMP(void)
{
    if(this->is_mp_initialized)
    {
        if(this->Reallocate__Thread(1_UZ) == false) {
            ERR(L"An error has been triggered from the "
                L"`Reallocate__Thread(1)` function.");
            return false;
        }

        this->cache_number_threads = this->number_threads = 1_UZ;

        this->is_mp_initialized = false;
    }

    return true;
}
}
