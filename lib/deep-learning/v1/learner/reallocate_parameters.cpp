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
#include "deep-learning/v1/mem/reallocate.hpp"

namespace DL::v1 {
bool Model::Reallocate__Parameter(size_t const number_parameters_received)
{
    if(this->total_parameters_allocated != 0u)
    {
        // Connection index.
        if(this->ptr_array_ptr_connections != nullptr)
        {
            void **tmp_ptr_array_ptr_connections(Mem::reallocate_ptofpt<void*, true>(this->ptr_array_ptr_connections,
                                                                                                                                       number_parameters_received,
                                                                                                                                       this->total_parameters_allocated));
            if(tmp_ptr_array_ptr_connections == nullptr) {
              ERR(L"An error has been triggered from the "
                  L"`reallocate_pointers_array_cpp<%zu>"
                  L"(ptr, %zu, %zu, true)` function.",
                  sizeof(void *), number_parameters_received,
                  this->total_parameters_allocated);
              return false;
            }
            this->ptr_array_ptr_connections = tmp_ptr_array_ptr_connections;
        }
        // |END| Connection index. |END|

        // Parameters.
        if(this->ptr_array_parameters != nullptr)
        {
            var *tmp_ptr_array_parameters(Mem::reallocate(this->ptr_array_parameters,
                                                                                                    number_parameters_received,
                                                                                                    this->total_parameters_allocated));
            this->ptr_array_parameters = tmp_ptr_array_parameters;
            
            if(this->Reallocate__Parameter__Optimizer(number_parameters_received) == false) {
              ERR(L"An error has been triggered from the "
                  L"`Reallocate__Parameter__Optimizer(%zu)` function.",
                  number_parameters_received);
                return false;
            }
            else if(this->Use__Regularization_Parameter() && this->Reallocate__Parameter__Regularization(number_parameters_received) == false) {
                ERR(L"An error has been triggered from the "
                    L"`Reallocate__Parameter__Regularization(%zu)` function.",
                    number_parameters_received);
                return false;
            }
            
            if(this->Use__Normalization()) { this->Reset__Parameter__Normalized_Unit(); }
        }
        // |END| Parameters. |END|

        // Derivates parameters.
        if(this->ptr_array_derivatives_parameters != nullptr)
        {
            real *tmp_ptr_array_derivatives_parameters(Mem::reallocate(this->ptr_array_derivatives_parameters,
                                                                                                                    this->number_threads * number_parameters_received,
                                                                                                                    this->number_threads * this->total_parameters_allocated));
            this->ptr_array_derivatives_parameters = tmp_ptr_array_derivatives_parameters;

            if(this->Use__Normalization()) { this->Reset__Derivative_Parameter__Normalized_Unit(); }
        }
        // |END| Derivates parameters. |END|

        this->total_parameters = number_parameters_received;
        this->total_parameters_allocated = number_parameters_received;
    }

    return true;
}

bool Model::Reallocate__Parameter__Regularization(size_t const number_parameters_received)
{
    if(this->ptr_array_mask_regularized_parameters != nullptr)
    {
        // Mask regularization parameters.
        real *tmp_ptr_array_mask_rergularization_parameters(Mem::reallocate(this->ptr_array_mask_regularized_parameters,
                                                                                                                                number_parameters_received,
                                                                                                                                this->total_parameters_allocated));
        memset(tmp_ptr_array_mask_rergularization_parameters + this->total_weights_allocated,
                       0,
               (number_parameters_received - this->total_weights_allocated) *
                   sizeof(real));

        this->ptr_array_mask_regularized_parameters = tmp_ptr_array_mask_rergularization_parameters;
        // |END| Mask regularization parameters. |END|
    }

    return true;
}

bool Model::Reallocate__Parameter__Optimizer(size_t const number_parameters_received)
{
    switch(this->type_optimizer_function)
    {
        case OPTIMIZER::GD: return(this->Reallocate__Parameter__Gradient_Descent(number_parameters_received));
        case OPTIMIZER::IRPROP_MINUS: return(this->Reallocate__Parameter__iRPROP_minus(number_parameters_received));
        case OPTIMIZER::IRPROP_PLUS: return(this->Reallocate__Parameter__iRPROP_plus(number_parameters_received));
        case OPTIMIZER::ADABOUND:
        case OPTIMIZER::ADAM:
        case OPTIMIZER::ADAMAX:
        case OPTIMIZER::NOSADAM: return(this->Reallocate__Parameter__Adam(number_parameters_received));
        case OPTIMIZER::AMSBOUND:
        case OPTIMIZER::AMSGRAD: return(this->Reallocate__Parameter__AMSGrad(number_parameters_received));
        default: return true;
    }
}

bool Model::Reallocate__Parameter__Gradient_Descent(size_t const number_parameters_received)
{
    if(this->learning_momentum != 0_r
      &&
      this->ptr_array_previous_delta_parameters != nullptr)
    {
        // Previous delta parameters.
        real *tmp_ptr_array_previous_delta_parameters(Mem::reallocate(
          this->ptr_array_previous_delta_parameters,
                                                                                                                      number_parameters_received,
                                                                                                                      this->total_parameters_allocated));
        this->ptr_array_previous_delta_parameters = tmp_ptr_array_previous_delta_parameters;
        // |END| Previous delta parameters. |END|
    }

    return true;
}

bool Model::Reallocate__Parameter__iRPROP_minus(size_t const number_parameters_received)
{
    if(this->ptr_array_previous_steps != nullptr)
    {
        real *tmp_ptr_array_previous_steps(Mem::reallocate(this->ptr_array_previous_steps,
                                                                                                     number_parameters_received,
                                                                                                     this->total_parameters_allocated));
        this->ptr_array_previous_steps = tmp_ptr_array_previous_steps;
        
        Mem::fill(this->ptr_array_previous_steps + this->total_weights_allocated,
                                  this->ptr_array_previous_steps + number_parameters_received,
                                  this->rprop_delta_zero);
    }
    
    if(this->ptr_array_previous_derivatives_parameters != nullptr)
    {
      real *tmp_ptr_array_previous_derivatives_parameters(Mem::reallocate(
          this->ptr_array_previous_derivatives_parameters,
                                                                                                                              number_parameters_received,
                                                                                                                              this->total_parameters_allocated));
        this->ptr_array_previous_derivatives_parameters = tmp_ptr_array_previous_derivatives_parameters;
    }

    return true;
}

bool Model::Reallocate__Parameter__iRPROP_plus(size_t const number_parameters_received)
{
    if(this->Reallocate__Parameter__iRPROP_minus(number_parameters_received) == false)
    {
        ERR(L"An error has been triggered from the "
            L"`Reallocate__Parameter__iRPROP_minus()` function.");
        return false;
    }

    if(this->ptr_array_previous_delta_parameters != nullptr)
    {
      real *tmp_ptr_array_previous_delta_parameters(Mem::reallocate(
          this->ptr_array_previous_delta_parameters,
                                                                                                                      number_parameters_received,
                                                                                                                      this->total_parameters_allocated));
        this->ptr_array_previous_delta_parameters = tmp_ptr_array_previous_delta_parameters;
    }

    return true;
}

bool Model::Reallocate__Parameter__Adam(size_t const number_parameters_received)
{
    if(this->ptr_array_previous_biased_first_moment != nullptr)
    {
    real *tmp_ptr_array_previous_biased_first_moment(Mem::reallocate(
        this->ptr_array_previous_biased_first_moment,
                                                                                                                           number_parameters_received,
                                                                                                                           this->total_parameters_allocated));
        this->ptr_array_previous_biased_first_moment = tmp_ptr_array_previous_biased_first_moment;
    }
    
    if(this->ptr_array_previous_biased_second_moment != nullptr)
    {
        real *tmp_ptr_array_previous_biased_second_moment(Mem::reallocate(this->ptr_array_previous_biased_second_moment,
                                                                                                                                number_parameters_received,
                                                                                                                                this->total_parameters_allocated));
        this->ptr_array_previous_biased_second_moment = tmp_ptr_array_previous_biased_second_moment;
    }

    return true;
}

bool Model::Reallocate__Parameter__AMSGrad(size_t const number_parameters_received)
{
    if(this->Reallocate__Parameter__Adam(number_parameters_received) == false)
    {
        ERR(L"An error has been triggered from the "
            L"`Reallocate__Parameter__Adam()` function.");
        return false;
    }

    if(this->ptr_array_previous_biased_second_moment_hat != nullptr)
    {
      real *tmp_ptr_array_previous_biased_second_moment_hat(
          Mem::reallocate(
              this->ptr_array_previous_biased_second_moment_hat,
                                                                                                                                      number_parameters_received,
                                                                                                                                      this->total_parameters_allocated));
        this->ptr_array_previous_biased_second_moment_hat = tmp_ptr_array_previous_biased_second_moment_hat;
    }

    return true;
}
}