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
struct LAYER {
  typedef enum : int {
    NONE = 0,
    AVERAGE_POOLING = 1,
    CONVOLUTION = 2,
    FULLY_CONNECTED = 3,
    FULLY_CONNECTED_INDEPENDENTLY_RECURRENT = 4,
    FULLY_CONNECTED_RECURRENT = 5,
    GRU = 6,
    LSTM = 7,
    MAX_POOLING = 8,
    RESIDUAL = 9,
    SHORTCUT = 10,
    LENGTH = 11
  } TYPE;
};

static std::map<LAYER::TYPE, std::wstring> LAYER_NAME = {
    {LAYER::NONE, L"NONE"},
    {LAYER::AVERAGE_POOLING, L"Average pooling"},
    {LAYER::CONVOLUTION, L"[x] Convolution"},
    {LAYER::FULLY_CONNECTED, L"Fully connected"},
    {LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT,
     L"Fully connected, independently recurrent"},
    {LAYER::FULLY_CONNECTED_RECURRENT, L"[x] Fully connected, recurrent"},
    {LAYER::GRU, L"[x] Gated recurrent unit"},
    {LAYER::LSTM, L"Long short-term memory"},
    {LAYER::MAX_POOLING, L"Max pooling"},
    {LAYER::RESIDUAL, L"Residual"},
    {LAYER::SHORTCUT, L"[x] Shorcut"},
    {LAYER::LENGTH, L"LENGTH"}};

static std::map<LAYER::TYPE, std::wstring> LAYER_CONN_NAME = {
    {LAYER::NONE, L"NONE"},
    {LAYER::AVERAGE_POOLING, L"connected_to_basic"},
    {LAYER::CONVOLUTION, L"connected_to_convolution"},
    {LAYER::FULLY_CONNECTED, L"connected_to_neuron"},
    {LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT, L"connected_to_AF_Ind_R"},
    {LAYER::FULLY_CONNECTED_RECURRENT, L"connected_to_neuron"},
    {LAYER::GRU, L"connected_to_cell"},
    {LAYER::LSTM, L"connected_to_cell"},
    {LAYER::MAX_POOLING, L"connected_to_basic_indice"},
    {LAYER::RESIDUAL, L"connected_to_basic"},
    {LAYER::SHORTCUT, L"connected_to_neuron"},
    {LAYER::LENGTH, L"LENGTH"}};
}  // namespace DL
