/* Copyright 2016, 2022 Sébastien Kéroack. All Rights Reserved.

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

// PCH:
#include "pch.hpp"

// File header:
#include "deep-learning/ops/modwt.hpp"

// Deep learning:
#include "deep-learning/io/logger.hpp"
#include "deep-learning/v1/mem/reallocate.hpp"

// Standard:
#include <cmath>

namespace DL::Math {
bool modwt(size_t const size_array, size_t &size_matrix,
           real const *const inputs, real *&outputs, size_t j_lvl) {
  size_t const max_j_lvl(
      static_cast<size_t>(log(static_cast<real>(size_array)) / log(2_r)));

  // |STR| Safety. |STR|
  if (j_lvl > max_j_lvl) {
    ERR(L"J level (%zu) greater than maximum level (%zu).", j_lvl, max_j_lvl);
    return false;
  }
  // |END| Safety. |END|

  if (j_lvl == 0_UZ) j_lvl = max_j_lvl;

  // |STR| Output. |STR|
  if (outputs != nullptr) {
    if (size_matrix != size_array * (j_lvl + 1_UZ))
      outputs = v1::Mem::reallocate<real, false, false>(
          outputs, size_array * (j_lvl + 1_UZ), size_matrix);
  } else
    outputs = new real[size_array * (j_lvl + 1_UZ)];

  size_matrix = size_array * (j_lvl + 1_UZ);
  // |END| Output. |END|

  real *approximations(new real[size_array]);
  memcpy(approximations, inputs, size_array * sizeof(real));

  // |STR| Filters. |STR|
  // High pass db1: {1 / sqrt(2) * 1, 1 / sqrt(2) * -1}
  // Low pass db1: {1 / sqrt(2), 1 / sqrt(2)}
  // real hih_pass_db1[2] =
  // {-7.071067811865475244008443621048490392848359376884740365883398e-01_r,
  // 7.071067811865475244008443621048490392848359376884740365883398e-01_r};
  real hih_pass_db1[2] = {
      7.071067811865475244008443621048490392848359376884740365883398e-01_r,
      -7.071067811865475244008443621048490392848359376884740365883398e-01_r};
  real low_pass_db1[2] = {
      7.071067811865475244008443621048490392848359376884740365883398e-01_r,
      7.071067811865475244008443621048490392848359376884740365883398e-01_r};

  hih_pass_db1[0] /= std::sqrt(2_r);
  hih_pass_db1[1] /= std::sqrt(2_r);
  low_pass_db1[0] /= std::sqrt(2_r);
  low_pass_db1[1] /= std::sqrt(2_r);
  // |END| Filters. |END|

  for (size_t j(0_UZ); j != j_lvl; ++j) {
    circular_convolve_decomposition(j + 1_UZ, size_array, 2_UZ, hih_pass_db1,
                                    approximations, outputs + j * size_array);

    circular_convolve_decomposition(j + 1_UZ, size_array, 2_UZ, low_pass_db1,
                                    approximations,
                                    outputs + (j + 1_UZ) * size_array);

    if (j + 1_UZ != j_lvl)
      memcpy(approximations, outputs + (j + 1_UZ) * size_array,
             size_array * sizeof(real));
  }

  delete[] (approximations);

  return true;
}

bool modwt_inverse(size_t const size_matrix_received, size_t const size_array,
                   real const *const inputs, real *&outputs, size_t j_lvl) {
  size_t const max_j_lvl(size_matrix_received / size_array - 1_UZ);
  size_t j_inv, j;

  // |STR| Safety. |STR|
  if (size_matrix_received == 0_UZ) {
    ERR(L"`size_matrix_received` can not be equal to zero.");
    return false;
  } else if (size_array == 0_UZ) {
    ERR(L"`size_array` can not be equal to zero.");
    return false;
  } else if (j_lvl > max_j_lvl) {
    ERR(L"J level (%zu) greater than allowable level (%zu).", j_lvl, max_j_lvl);
    return false;
  }

  if (j_lvl == 0_UZ)
    j_lvl = max_j_lvl;
  else
    j_lvl = max_j_lvl - j_lvl;
  // |END| Safety. |END|

  // |STR| Output. |STR|
  if (outputs != nullptr)
    outputs =
        v1::Mem::reallocate<real, false, false>(outputs, size_array, 0_UZ);
  else
    outputs = new real[size_array];

  memset(outputs, 0, size_array * sizeof(real));
  // |END| Output. |END|

  real *approximations(new real[size_array]);
  memcpy(approximations, inputs + max_j_lvl * size_array,
         size_array * sizeof(real));

  // |STR| Filters. |STR|
  // High pass db1/haar: {1 / sqrt(2) * 1, 1 / sqrt(2) * -1}
  // Low pass db1/haar: {1 / sqrt(2), 1 / sqrt(2)}
  // real hih_pass_db1[2] =
  // {-7.071067811865475244008443621048490392848359376884740365883398e-01_r,
  // 7.071067811865475244008443621048490392848359376884740365883398e-01_r};
  real hih_pass_db1[2] = {
      7.071067811865475244008443621048490392848359376884740365883398e-01_r,
      -7.071067811865475244008443621048490392848359376884740365883398e-01_r};
  real low_pass_db1[2] = {
      7.071067811865475244008443621048490392848359376884740365883398e-01_r,
      7.071067811865475244008443621048490392848359376884740365883398e-01_r};

  hih_pass_db1[0] /= std::sqrt(2_r);
  hih_pass_db1[1] /= std::sqrt(2_r);
  low_pass_db1[0] /= std::sqrt(2_r);
  low_pass_db1[1] /= std::sqrt(2_r);
  // |END| Filters. |END|

  for (j = 0_UZ; j != j_lvl; ++j) {
    j_inv = max_j_lvl - j - 1_UZ;

    circular_convolve_reconstruction(
        j_inv + 1_UZ, size_array, 2_UZ, hih_pass_db1, low_pass_db1,
        inputs + j_inv * size_array, approximations, outputs);

    if (j + 1_UZ != j_lvl)
      memcpy(approximations, outputs, size_array * sizeof(real));
  }

  delete[] (approximations);

  return true;
}

void circular_convolve_decomposition(size_t const j_lvl,
                                     size_t const size_inputs,
                                     size_t const size_filters,
                                     real const *const filters_pass,
                                     real const *const inputs,
                                     real *const outputs) {
  size_t t, i;

  real sum;

  for (t = 0_UZ; t != size_inputs; ++t) {
    sum = 0_r;

    for (i = 0_UZ; i != size_filters; ++i)
      sum += inputs[(t + size_inputs -
                     static_cast<size_t>(
                         pow(2.0, static_cast<double>(j_lvl) - 1.0)) *
                         i) %
                    size_inputs] *
             filters_pass[i];

    outputs[t] = sum;
  }
}

void circular_convolve_reconstruction(
    size_t const j_lvl, size_t const size_inputs, size_t const size_filters,
    real const *const filters_high_pass, real const *const filters_low_pass,
    real const *const inputs, real const *const inputs_tm1,
    real *const outputs) {
  size_t t, i;

  real sum;

  for (t = 0_UZ; t != size_inputs; ++t) {
    sum = 0_r;

    for (i = 0_UZ; i != size_filters; ++i) {
      sum += inputs[(t + static_cast<size_t>(
                             pow(2.0, static_cast<double>(j_lvl) - 1.0)) *
                             i) %
                    size_inputs] *
             filters_high_pass[i];

      sum += inputs_tm1[(t + static_cast<size_t>(
                                 pow(2.0, static_cast<double>(j_lvl) - 1.0)) *
                                 i) %
                        size_inputs] *
             filters_low_pass[i];
    }

    outputs[t] = sum;
  }
}
}  // namespace DL::Math