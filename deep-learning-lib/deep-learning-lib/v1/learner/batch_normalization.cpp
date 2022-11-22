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

namespace DL::v1 {
bool Model::Set__Normalization_Momentum_Average(real const momentum_average_received)
{
    if(this->normalization_momentum_average == momentum_average_received) { return true; }

    this->normalization_momentum_average = momentum_average_received;

#ifdef COMPILE_CUDA
    if(this->is_cu_initialized)
    { this->cumodel->Set__Normalization_Momentum_Average(momentum_average_received); }
#endif

    return true;
}

bool Model::Set__Normalization_Epsilon(real const epsilon_received) {
    if(this->normalization_epsilon == epsilon_received) { return true; }

    this->normalization_epsilon = epsilon_received;

#ifdef COMPILE_CUDA
    if(this->is_cu_initialized)
    { this->cumodel->Set__Normalization_Epsilon(epsilon_received); }
#endif

    return true;
}
}
