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
#include "deep-learning-lib/v1/mem/reallocate.hpp"

namespace DL::v1 {
void Model::reset_global_loss(void) {
  this->loss_train = HUGE_VAL;
  this->loss_valid = HUGE_VAL;
  this->loss_testg = HUGE_VAL;

  this->acc_train = 0.0;
  this->acc_valid = 0.0;
  this->acc_testg = 0.0;
}

void Model::reset_loss(void)
{
#ifdef COMPILE_CUDA
    if(this->is_cu_initialized)
    { this->cumodel->reset_loss(); }
    else
#endif
    {
        this->n_acc_trial = 0_UZ;

        if(this->ptr_array_number_bit_fail != nullptr)
        {
            memset(this->ptr_array_number_bit_fail,
                        0,
                        this->number_threads * sizeof(size_t));
        }
        
        if(this->ptr_array_number_loss != nullptr)
        {
            memset(this->ptr_array_number_loss,
                        0,
                        this->number_threads * sizeof(size_t));
        }
        
        if(this->ptr_array_loss_values != nullptr)
        {
          memset(this->ptr_array_loss_values,
                        0,
                        this->number_threads * sizeof(double));
        }
        
        if(this->ptr_array_accuracy_values[0] != nullptr)
        {
          memset(this->ptr_array_accuracy_values[0],
                        0,
                 this->number_threads * sizeof(double));
        }
        
        if(this->ptr_array_accuracy_values[1] != nullptr)
        {
          memset(this->ptr_array_accuracy_values[1],
                        0,
                 this->number_threads * sizeof(double));
        }
        
        if(this->ptr_array_accuracy_values[2] != nullptr)
        {
          memset(this->ptr_array_accuracy_values[2],
                        0,
                 this->number_threads * sizeof(double));
        }
        
        if(this->ptr_array_accuracy_values[3] != nullptr)
        {
          memset(this->ptr_array_accuracy_values[3],
                        0,
                 this->number_threads * sizeof(double));
        }
        
        if(this->ptr_array_accuracy_values[4] != nullptr)
        {
          memset(this->ptr_array_accuracy_values[4],
                        0,
                 this->number_threads * sizeof(double));
        }
    }
}
    
double Model::get_loss(ENV::TYPE const env) const {
  double loss;

    switch (env)
    {
        case ENV::TRAIN: loss = this->loss_train; break;
        case ENV::VALID: loss = this->loss_valid; break;
        case ENV::TESTG: loss = this->loss_testg; break;
        case ENV::NONE:
            switch(this->type_loss_function)
            {
                case LOSS_FN::ME: loss = this->get_me(); break;
                case LOSS_FN::L1: loss = this->get_loss_l1(); break;
                case LOSS_FN::MAE: loss = this->get_mae(); break;
                case LOSS_FN::L2: loss = this->get_loss_l2(); break;
                case LOSS_FN::MSE: loss = this->get_mse(); break;
                case LOSS_FN::RMSE: loss = this->get_rmse(); break;
                case LOSS_FN::MAPE: loss = this->get_mape(); break;
                case LOSS_FN::SMAPE: loss = this->get_smape(); break;
                case LOSS_FN::MASE_SEASONAL: loss = this->get_mase(); break;
                case LOSS_FN::MASE_NON_SEASONAL: loss = this->get_mase(); break;
                case LOSS_FN::CROSS_ENTROPY: loss = this->get_ace(); break;
                case LOSS_FN::BIT: loss = this->get_bitfail(); break;
                default: loss = 1.0; break;
            }
                break;
        default: loss = 1.0; break;
    }

    return loss;
}
    
double Model::get_me(
    void) const  // https://en.wikipedia.org/wiki/Mean_absolute_error
{
    if(*this->ptr_array_number_loss != 0_UZ)
      return *this->ptr_array_loss_values / static_cast<double>(*this->ptr_array_number_loss);
    else
      return 1.0;
}
    
double Model::get_loss_l1(void) const { return *this->ptr_array_loss_values; }
    
double Model::get_mae(
    void) const  // https://en.wikipedia.org/wiki/Mean_absolute_error
{
  if (*this->ptr_array_number_loss != 0_UZ)
    return *this->ptr_array_loss_values /
           static_cast<double>(*this->ptr_array_number_loss);
  else
    return 1.0;
}
    
double Model::get_loss_l2(void) const { return *this->ptr_array_loss_values; }
    
double Model::get_mse(
    void) const  // https://en.wikipedia.org/wiki/Mean_squared_error
{
    if(*this->ptr_array_number_loss != 0_UZ)
     return 1.0 / static_cast<double>(*this->ptr_array_number_loss) * *this->ptr_array_loss_values;
    else
     return 1.0;
}
    
double Model::get_rmse(
    void) const  // https://en.wikipedia.org/wiki/Root-mean-square_deviation
{
  if (*this->ptr_array_number_loss != 0_UZ)
    return sqrt(1.0 / static_cast<double>(*this->ptr_array_number_loss) *
                *this->ptr_array_loss_values);
  else
    return 1.0;
}
    
double Model::get_mape(
    void) const  // https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
{
  if (*this->ptr_array_number_loss != 0_UZ)
    return 1.0 / static_cast<double>(*this->ptr_array_number_loss) *
           *this->ptr_array_loss_values;
  else
    return 1.0;
}
    
double Model::get_smape(
    void) const  // https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
{
  if (*this->ptr_array_number_loss != 0_UZ)
    return 1.0 / static_cast<double>(*this->ptr_array_number_loss) *
           *this->ptr_array_loss_values;
  else
    return 1.0;
}

[[deprecated("Not properly implemented.")]] double Model::get_mase(
    void) const  // https://en.wikipedia.org/wiki/Mean_absolute_scaled_error
{
    // Non seasonal time series
    //if(*this->ptr_array_number_loss != 0_UZ
    //   &&
    //   this->mean_absolute_error_denominator != 0.0f
    //   &&
    //   *this->ptr_array_number_loss > 1u)
    //{ return(*this->ptr_array_loss_values / ((static_cast<var>(*this->ptr_array_number_loss) / static_cast<var>(*this->ptr_array_number_loss - 1u)) * this->mean_absolute_error_denominator)); }
    //{ return(1_r / *this->ptr_array_number_loss * (*this->ptr_array_loss_values / ((1_r / static_cast<var>(*this->ptr_array_number_loss - 1u)) * this->mean_absolute_error_denominator))); }
    //{ return(1_r / this->seq_w * (*this->ptr_array_loss_values / ((1_r / static_cast<var>(this->seq_w - 1_UZ)) * this->mean_absolute_error_denominator))); }
    /*else*/    { return 1.0; }
}

double Model::get_ace(
    void) const  // https://en.wikipedia.org/wiki/Cross_entropy
{
  if (*this->ptr_array_number_loss != 0_UZ)
    return *this->ptr_array_loss_values /
           static_cast<double>(*this->ptr_array_number_loss /
                               this->get_n_out());
  else
    return HUGE_VAL;
}

double Model::get_bitfail(void) const  // link
{ return static_cast<double>(*this->ptr_array_number_bit_fail); }
    
void Model::set_loss(ENV::TYPE const env, double const loss) {
    switch(env)
    {
        case ENV::TRAIN: this->loss_train = loss; break;
        case ENV::VALID: this->loss_valid = loss; break;
        case ENV::TESTG: this->loss_testg = loss; break;
        default:
            ERR(L"Loss type (%d) is not managed in", env);
                break;
    }
}
}
