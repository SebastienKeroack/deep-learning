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
// TODO: Remove this enumeration.
enum class ENUM_TYPE_DATASET_MANAGER_STORAGE : unsigned int {
  TYPE_STORAGE_NONE = 0,
  TYPE_STORAGE_TRAINING = 1,
  TYPE_STORAGE_TRAINING_TESTING = 2,
  TYPE_STORAGE_TRAINING_VALIDATION_TESTING = 3,
  TYPE_STORAGE_LENGTH = 4u
};

// TODO: Remove this enumeration.
static std::map<enum ENUM_TYPE_DATASET_MANAGER_STORAGE, std::wstring>
    ENUM_TYPE_DATASET_MANAGER_STORAGE_NAMES = {
        {ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_NONE, L"NONE"},
        {ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING, L"Train"},
        {ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_TRAINING_TESTING,
         L"Train - Testg"},
        {ENUM_TYPE_DATASET_MANAGER_STORAGE::
             TYPE_STORAGE_TRAINING_VALIDATION_TESTING,
         L"Train - Valid - Testg"},
        {ENUM_TYPE_DATASET_MANAGER_STORAGE::TYPE_STORAGE_LENGTH, L"LENGTH"}};

struct DATA {
  typedef enum : int {
    INPUT = 0,
    OUTPUT = 1,
  } TYPE;
};

struct DATASET_FORMAT {
  typedef enum : int {
    STANDARD = 0,
    SPLIT = 1,
    MNIST = 2,  // National Institute of Standards and Technology (NIST)
    LENGTH = 3
  } TYPE;
};

static std::map<DATASET_FORMAT::TYPE, std::wstring> DATASET_FORMAT_NAME = {
    {DATASET_FORMAT::STANDARD, L"DatasetV1"},
    {DATASET_FORMAT::SPLIT, L"DatasetV1 split"},
    {DATASET_FORMAT::MNIST, L"MNIST"},
    {DATASET_FORMAT::LENGTH, L"LENGTH"}};

struct DATASET {
  typedef enum : int {
    NONE = 0,
    BATCH = 1,
    MINIBATCH = 2,
    CROSS_VAL = 3,
    CROSS_VAL_OPT = 4,
    LENGTH = 5
  } TYPE;
};

static std::map<DATASET::TYPE, std::wstring> DATASET_NAME = {
    {DATASET::NONE, L"NONE"},
    {DATASET::BATCH, L"Batch"},
    {DATASET::MINIBATCH, L"Mini batch"},
    {DATASET::CROSS_VAL, L"Cross validation"},
    {DATASET::CROSS_VAL_OPT, L"Cross validation, random search"},
    {DATASET::LENGTH, L"LENGTH"}};
}  // namespace DL
