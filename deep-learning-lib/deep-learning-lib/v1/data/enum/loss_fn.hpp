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

#include <string>
#include <map>

namespace DL::v1 {
struct LOSS_FN {
  typedef enum : int {
    NONE = 0,
    BIT = 1,
    CROSS_ENTROPY = 2,  // Cross entropy.
    L1 = 3,             // L1, Least absolute deviation.
    L2 = 4,             // L2, Least squares error.
    MAE = 5,            // Mean absolute error.
    MAPE = 6,           // Mean absolute percentage error.
    MASE_NON_SEASONAL =
        7,  // Mean absolute scaled error, non seasonal time series.
    MASE_SEASONAL = 8,  // Mean absolute scaled error, seasonal time series.
    ME = 9,             // Mean error.
    MSE = 10,           // Mean square error.
    RMSE = 11,          // Root mean square error.
    SMAPE = 12,         // Symmetric mean absolute percentage error.
    LENGTH = 13
  } TYPE;
};

static std::map<LOSS_FN::TYPE, std::wstring> LOSS_FN_NAME = {
    {LOSS_FN::NONE, L"NONE"},
    {LOSS_FN::BIT, L"Bit"},
    {LOSS_FN::CROSS_ENTROPY, L"Cross-entropy"},
    {LOSS_FN::L1, L"[-] L1"},
    {LOSS_FN::L2, L"L2"},
    {LOSS_FN::MAE, L"Mean absolute error"},
    {LOSS_FN::MAPE, L"Mean absolute percentage error"},
    {LOSS_FN::MASE_NON_SEASONAL,
     L"[x] Mean absolute scaled error, non seasonal time series"},
    {LOSS_FN::MASE_SEASONAL,
     L"[x] Mean absolute scaled error, seasonal time series"},
    {LOSS_FN::ME, L"[-] Mean error"},
    {LOSS_FN::MSE, L"Mean square error"},
    {LOSS_FN::RMSE, L"Root mean square error"},
    {LOSS_FN::SMAPE, L"Symmetric mean absolute percentage error"},
    {LOSS_FN::LENGTH, L"LENGTH"}};

struct ACCU_FN {
  typedef enum : int {
    NONE = 0,
    CROSS_ENTROPY = 1,
    DIRECTIONAL = 2,
    DISTANCE = 3,
    // https://en.wikipedia.org/wiki/Correlation_coefficient | Pearson
    R = 4,
    SIGN = 5,
    LENGTH = 6
  } TYPE;
};

static std::map<ACCU_FN::TYPE, std::wstring> ACC_FN_NAME = {
    {ACCU_FN::NONE, L"NONE"},
    {ACCU_FN::CROSS_ENTROPY, L"Cross-entropy"},
    {ACCU_FN::DIRECTIONAL, L"Directional"},
    {ACCU_FN::DISTANCE, L"Distance"},
    {ACCU_FN::R, L"R"},
    {ACCU_FN::SIGN, L"sign"},
    {ACCU_FN::LENGTH, L"LENGTH"}};
}  // namespace DL
