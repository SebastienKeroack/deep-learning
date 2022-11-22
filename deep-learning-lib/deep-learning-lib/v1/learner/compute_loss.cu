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

#include "deep-learning-lib/v1/learner/model.cuh"

__device__ void cuModel::Test(size_t const batch_size,
                                                          var **const ptr_array_outputs_received,
                                                          size_t const time_step_index_received)
{ this->FF__Test(batch_size, ptr_array_outputs_received); }

__device__ void cuModel::FF__Test(size_t const batch_size, var **const ptr_array_outputs_received)
{
    switch(this->type_loss_function)
    {
        case DL::LOSS_FN::ME:
        case DL::LOSS_FN::L1:
        case DL::LOSS_FN::MAE:
        case DL::LOSS_FN::L2:
        case DL::LOSS_FN::MSE:
        case DL::LOSS_FN::RMSE:
        case DL::LOSS_FN::MAPE:
        case DL::LOSS_FN::SMAPE:
        case DL::LOSS_FN::MASE_SEASONAL:
        case DL::LOSS_FN::MASE_NON_SEASONAL:
            this->FF__Test__Standard(batch_size, ptr_array_outputs_received);
                break;
        case DL::LOSS_FN::CROSS_ENTROPY:
            this->FF__Test__Binary_Cross_Entropy(batch_size, ptr_array_outputs_received);
                break;
        case DL::LOSS_FN::BIT: this->FF__Test__Bit_Fail(batch_size, ptr_array_outputs_received); break;
        default:
            ERR(L"Undefined type loss function (%u).",
                                    this->type_loss_function);
                break;
    }
}

__device__ void cuModel::RNN__Test(size_t const batch_size,
                                                                    var **const ptr_array_outputs_received,
                                                                    size_t const time_step_index_received)
{
}
