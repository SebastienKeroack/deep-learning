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

namespace DL::Math {
bool modwt(size_t const size_array, size_t &size_matrix,
           real const *const inputs, real *&outputs, size_t j_lvl = 0_UZ);

bool modwt_inverse(size_t const size_matrix_received, size_t const size_array,
                   real const *const inputs, real *&outputs,
                   size_t j_lvl = 0_UZ);

void circular_convolve_decomposition(size_t const j_lvl,
                                     size_t const size_inputs,
                                     size_t const size_filters,
                                     real const *const filters_pass,
                                     real const *const inputs,
                                     real *const outputs);

void circular_convolve_reconstruction(
    size_t const j_lvl, size_t const size_inputs, size_t const size_filters,
    real const *const filters_high_pass, real const *const filters_low_pass,
    real const *const inputs, real const *const inputs_tm1,
    real *const outputs);
}  // namespace DL::Math