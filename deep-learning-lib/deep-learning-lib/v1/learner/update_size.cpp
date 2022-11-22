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

#include "deep-learning-lib/pch.hpp"

#include "deep-learning-lib/v1/learner/model.hpp"
#include "deep-learning-lib/data/string.hpp"
#include "deep-learning-lib/data/time.hpp"
#include "deep-learning-lib/io/logger.hpp"

#include <omp.h>

namespace DL::v1 {
bool Model::Allouable__Batch_Size(size_t const desired_batch_size_received,
                                                                  size_t &ref_batch_size_allouable_received,
                                                                  size_t &ref_number_threads_allouable_received)
{
    // Size of a thread.
    size_t const tmp_size_thread(this->Get__Threads_Sizeof(1_UZ)),
    // Size of a batch.
                       tmp_size_batch(this->Get__Batch_Sizeof(1_UZ)),
    // Size of a neural network with no batch.
                       tmp_size_neural_network(this->Get__Sizeof(1_UZ, 1_UZ) - (tmp_size_thread + tmp_size_batch)),
    // Available memory substraction size of the neural network without batch.
                       tmp_available_memory_mbs(this->maximum_allowable_memory_bytes - tmp_size_neural_network);
    
    // If the neural network overflow the maximum allowable memory.
    if(this->maximum_allowable_memory_bytes < tmp_size_neural_network)
    {
        ERR(L"Maximum allowable memory (%zu) is less than the memory allocate (%zu).",
                                 this->maximum_allowable_memory_bytes,
                                 tmp_size_neural_network + tmp_size_thread + tmp_size_batch);

        ref_batch_size_allouable_received = 0_UZ;
        ref_number_threads_allouable_received = 0_UZ;

        return false;
    }
    // If one the size of one thread overflow the available memory.
    else if(tmp_available_memory_mbs < tmp_size_thread)
    {
        ERR(L"No enought available memory (%zu) for allocating %zu bytes.",
                                 tmp_available_memory_mbs,
                                 tmp_size_thread);

        ref_batch_size_allouable_received = 0_UZ;
        ref_number_threads_allouable_received = 0_UZ;

        return false;
    }
    // If one the size of one batch overflow the available memory.
    else if(tmp_available_memory_mbs - tmp_size_thread < tmp_size_batch)
    {
        ERR(L"No enought available memory (%zu) for allocating %zu bytes.",
                                 tmp_available_memory_mbs - tmp_size_thread,
                                 tmp_size_batch);

        ref_batch_size_allouable_received = 0_UZ;
        ref_number_threads_allouable_received = 0_UZ;

        return false;
    }

    size_t const tmp_maximum_threads((this->use_mp || this->is_mp_initialized) ? (this->pct_threads == 0.0f ? 1_UZ : std::min<size_t>(static_cast<size_t>(ceil(static_cast<double>(this->pct_threads) * static_cast<double>(omp_get_num_procs()) / 100.0)), desired_batch_size_received)) : 1_UZ);
    size_t tmp_maximum_batch_size_allocatable((tmp_available_memory_mbs - tmp_size_thread) / tmp_size_batch),
              tmp_batch_size_allocate(std::min<size_t>(desired_batch_size_received, this->maximum_batch_size)),
              tmp_threads_allocate(1_UZ);

    if(tmp_batch_size_allocate > tmp_maximum_batch_size_allocatable)
    {
        WARN(L"Can not allocate the optimal batch size (%zu). The batch size allocated will be reduced to %zu.",
                                 tmp_batch_size_allocate,
                                 tmp_maximum_batch_size_allocatable);

        // Batch size equal maximum batch size allocatables.
        tmp_batch_size_allocate = tmp_maximum_batch_size_allocatable;
    }
    else
    {
        for(; tmp_threads_allocate != tmp_maximum_threads; ++tmp_threads_allocate)
        {
            // Maximum batch size equal available memory minus allocates threads, then divide by one batch size.
            tmp_maximum_batch_size_allocatable = static_cast<size_t>((tmp_available_memory_mbs - tmp_threads_allocate * tmp_size_thread) / tmp_size_batch);

            // If batch size is greater than maximum batch size allocatables.
            if(tmp_batch_size_allocate > tmp_maximum_batch_size_allocatable)
            {
                WARN(L"Can not allocate the optimal number of threads (%zu). The number of threads allocated will be reduced to %zu.",
                                         tmp_threads_allocate,
                                         tmp_threads_allocate - 1_UZ);

                // Batch size equal available memory minus past allocates threads, then divide by one batch size.
                tmp_batch_size_allocate = static_cast<size_t>((tmp_available_memory_mbs - (tmp_threads_allocate - 1_UZ) * tmp_size_thread) / tmp_size_batch);

                break;
            }
        }
    }

    ref_batch_size_allouable_received = tmp_batch_size_allocate;
    ref_number_threads_allouable_received = tmp_threads_allocate;

    return true;
}

bool Model::update_mem_thread_size(size_t const desired_number_threads_received)
{
    if(this->is_mp_initialized == false) { return true; }
    else if(desired_number_threads_received <= this->cache_number_threads && this->pct_threads == this->pct_threads_cached) { return true; }
    
    size_t tmp_batch_size_allocate(desired_number_threads_received),
              tmp_number_threads_allocate(desired_number_threads_received);
    
    if(this->Allouable__Batch_Size(desired_number_threads_received,
                                                 tmp_batch_size_allocate,
                                                 tmp_number_threads_allocate) == false)
    {
        ERR(L"An error has been triggered from the \"Allouable__Thread_Size(%zu, %zu, %zu)\" function.",
                                 desired_number_threads_received,
                                 tmp_batch_size_allocate,
                                 tmp_number_threads_allocate);

        return false;
    }

    // If number of threads differ from the new desired.
    if(this->number_threads != tmp_number_threads_allocate)
    {
        if(this->Reallocate__Thread(tmp_number_threads_allocate) == false)
        {
            ERR(L"An error has been triggered from the \"Reallocate__Thread(%zu)\" function.",
                                     tmp_number_threads_allocate);

            return false;
        }

        if(this->update_mem_batch_size(this->cache_batch_size, true) == false)
        {
            ERR(L"An error has been triggered from the \"update_mem_batch_size(%zu, true)\" function.",
                                     this->cache_batch_size);

            return false;
        }
        
        this->number_threads = tmp_number_threads_allocate;
    }

    // If number of threads is greater than the cached.
    if(desired_number_threads_received > this->cache_number_threads) { this->cache_number_threads = desired_number_threads_received; }

    // Cache the maximum threads in percent.
    this->pct_threads_cached = this->pct_threads;

    return true;
}

bool Model::update_mem_batch_size(size_t const desired_batch_size_received, bool const force_update_received)
{
    if(force_update_received == false && desired_batch_size_received <= this->cache_batch_size) { return true; }
    
    size_t tmp_batch_size_allocate(desired_batch_size_received),
              tmp_number_threads_allocate(desired_batch_size_received);
    
    if(this->Allouable__Batch_Size(desired_batch_size_received,
                                                 tmp_batch_size_allocate,
                                                 tmp_number_threads_allocate) == false)
    {
        ERR(L"An error has been triggered from the \"Allouable__Thread_Size(%zu, %zu, %zu)\" function.",
                                 desired_batch_size_received,
                                 tmp_batch_size_allocate,
                                 tmp_number_threads_allocate);

        return false;
    }

    // If total data batch differ from the new desired
    if(this->batch_size != tmp_batch_size_allocate)
    {
        // reallocate batch size with the new batch size meet.
        if(this->Reallocate__Batch(tmp_batch_size_allocate) == false)
        {
            ERR(L"An error has been triggered from the \"Reallocate__Batch(%zu)\" function.",
                                     tmp_batch_size_allocate);

            return false;
        }

        this->batch_size = tmp_batch_size_allocate;
    }

    // Cache total data batch.
    this->cache_batch_size = desired_batch_size_received;

    return true;
}
}
