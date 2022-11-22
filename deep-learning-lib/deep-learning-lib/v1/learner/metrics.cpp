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
#include "deep-learning-lib/ops/math.hpp"

using namespace DL::Math;

namespace DL::v1 {
bool Model::Set__Accurancy_Variance(double const accurancy_variance_received)
{
    if(this->acc_var == accurancy_variance_received) { return true; }
    else if(accurancy_variance_received < 0.0)
    {
        ERR(L"Accuracy variance (%f) less than zero.",
                                 accurancy_variance_received);

        return false;
    }
    else if(accurancy_variance_received > 1.0)
    {
        ERR(L"Accuracy variance (%f) bigger than one.",
                                 accurancy_variance_received);

        return false;
    }

    this->acc_var = accurancy_variance_received;
    
#ifdef COMPILE_CUDA
    if(this->is_cu_initialized)
    { this->cumodel->Set__Accurancy_Variance(accurancy_variance_received); }
#endif

    return true;
}

bool Model::set_seq_w(size_t const time_delays_received)
{
    if(this->n_time_delay == time_delays_received) { return true; }
    else if(time_delays_received > this->seq_w)
    {
        ERR(L"Time delays (%zu) bigger than recurrent depth (%zu).",
                                 time_delays_received,
                                 this->seq_w);

        return false;
    }

    this->n_time_delay = time_delays_received;
    
#ifdef COMPILE_CUDA
    if(this->is_cu_initialized)
    { this->cumodel->set_seq_w(time_delays_received); }
#endif

    return true;
}

void Model::set_accu(ENV::TYPE const env_type, double const accurancy_received)
{
    switch(env_type)
    {
        case ENV::TRAIN: this->acc_train = accurancy_received; break;
        case ENV::VALID: this->acc_valid = accurancy_received; break;
        case ENV::TESTG: this->acc_testg = accurancy_received; break;
        default:
            ERR(L"Accuracy type (%d) is not managed in",
                                     env_type);
                break;
    }
}

double Model::get_accu(ENV::TYPE const env_type) const
{
    double tmp_accurancy;

    switch(env_type)
    {
        case ENV::TRAIN: tmp_accurancy = this->acc_train; break;
        case ENV::VALID: tmp_accurancy = this->acc_valid; break;
        case ENV::TESTG: tmp_accurancy = this->acc_testg; break;
        case ENV::NONE:
            switch(this->type_accuracy_function)
            {
                case ACCU_FN::R: tmp_accurancy = clip(this->ptr_array_accuracy_values[0][0], -1.0, 1.0); break; // Real-precession clip.
                default: tmp_accurancy = this->n_acc_trial == 0_UZ ? 0.0 : this->ptr_array_accuracy_values[0][0] / static_cast<double>(this->n_acc_trial) * 100.0; break;
            }
                break;
        default:
            ERR(L"Accuracy type (%d) is not managed in",
                                     env_type);

            tmp_accurancy = 0.0;
                break;
    }

    return(tmp_accurancy);
}
}
