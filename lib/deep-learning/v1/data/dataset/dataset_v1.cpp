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

#include "deep-learning/v1/data/datasets.hpp"
#include "deep-learning/v1/data/scaler.hpp"
#include "deep-learning/data/string.hpp"
#include "deep-learning/data/time.hpp"
#include "deep-learning/io/file.hpp"
#include "deep-learning/io/logger.hpp"
#include "deep-learning/io/term/input.hpp"
#include "deep-learning/ops/modwt.hpp"
#include "deep-learning/v1/mem/reallocate.hpp"

#include <fstream>
#include <iostream>
#include <omp.h>

using namespace DL::File;
using namespace DL::Str;
using namespace DL::Term;

namespace DL::v1 {
bool scan_datasets(DATASET_FORMAT::TYPE &type, std::wstring const &path_name) {
  std::vector<DATASET_FORMAT::TYPE> types;

  if (path_exist(path_name + L".dataset"))
    types.push_back(DATASET_FORMAT::STANDARD);

  if (path_exist(path_name + L".dataset-input") &&
      path_exist(path_name + L".dataset-output"))
    types.push_back(DATASET_FORMAT::SPLIT);

  if (path_exist(path_name + L".idx3-ubyte") &&
      path_exist(path_name + L".idx1-ubyte"))
    types.push_back(DATASET_FORMAT::MNIST);

  if (types.empty()) {
    ERR(L"No data available in the folder.");
    return false;
  }

  if (types.size() == 1_UZ) {
    type = types.at(0);
    return true;
  }

  INFO(L"");
  INFO(L"DatasetV1 file type available:");

  for (size_t length(types.size()), i(0_UZ); i != length; ++i)
    INFO(L"[%zu] Type: %ls.", i, DATASET_FORMAT_NAME[types.at(i)].c_str());

  type = types.at(parse_discrete(0, static_cast<int>(types.size()) - 1,
                                 L"Type data file: "));

  return true;
}

DatasetV1::DatasetV1(void) {}

DatasetV1::DatasetV1(DATASET_FORMAT::TYPE const dset_fmt,
                 std::wstring const &path_name) {
  this->allocate(dset_fmt, path_name);
}

DatasetV1::DatasetV1(DATASET_FORMAT::TYPE const dset_fmt,
                 ENV::TYPE const env_type,
                 std::wstring const &path_name)
    : env_type(env_type) {
  this->allocate(dset_fmt, path_name);
}

[[deprecated("Not properly implemented.")]] DatasetV1 &DatasetV1::operator=(
    DatasetV1 const &cls) {
  return *this;
}

[[deprecated("Not properly implemented?")]] void DatasetV1::copy(
    DatasetV1 const &cls) {
  if (this->_reference == false) this->Deallocate();

  this->_use_multi_label = cls._use_multi_label;

  this->p_n_data_alloc = this->p_n_data = cls.p_n_data;
  this->p_n_inp = cls.p_n_inp;
  this->p_n_out = cls.p_n_out;
  this->p_seq_w = cls.p_seq_w;

  this->p_file_buffer_size = cls.p_file_buffer_size;
  this->p_file_buffer_shift_size = cls.p_file_buffer_shift_size;

  this->Xm = cls.Xm;
  this->Ym = cls.Ym;

  this->p_type_dataset_process = cls.p_type_dataset_process;
  this->p_type_data_file = cls.p_type_data_file;

  this->_reference = true;

  this->p_str_i = cls.p_str_i;
}

void DatasetV1::reference(size_t const number_examples_received, real const **Xm,
                        real const **Ym, DatasetV1 const &cls) {
  this->Deallocate();

  this->_use_multi_label = cls.Use__Multi_Label();

  this->p_n_data_alloc = this->p_n_data = number_examples_received;
  this->p_n_inp = cls.get_n_inp();
  this->p_n_out = cls.get_n_out();
  this->p_seq_w = cls.get_seq_w();

  this->p_file_buffer_size = cls.p_file_buffer_size;
  this->p_file_buffer_shift_size = cls.p_file_buffer_shift_size;

  this->Xm = Xm;
  this->Ym = Ym;

  this->p_type_data_file = cls.p_type_data_file;

  this->_reference = true;

  this->p_str_i = 0_UZ;
}

void DatasetV1::Train_Epoch_OpenMP(Model *const model) {
  if (model->Use__Dropout__Bernoulli() ||
      model->Use__Dropout__Bernoulli__Inverted() ||
      model->Use__Dropout__Alpha()) {
    model->Dropout_Bernoulli();
  } else if (model->Use__Dropout__Zoneout()) {
    model->Dropout_Zoneout();
  }

  model->reset_loss();

  switch (model->type_optimizer_function) {
    case OPTIMIZER::GD:
    case OPTIMIZER::QUICKPROP:
    case OPTIMIZER::SARPROP:
    case OPTIMIZER::ADABOUND:
    case OPTIMIZER::ADAM:
    case OPTIMIZER::ADAMAX:
    case OPTIMIZER::AMSBOUND:
    case OPTIMIZER::AMSGRAD:
    case OPTIMIZER::NOSADAM:
      this->Train_Batch_BP_OpenMP(model);
      break;
    case OPTIMIZER::IRPROP_MINUS:
    case OPTIMIZER::IRPROP_PLUS:
      model->loss_rprop_tm1 = model->loss_rprop;

      this->Train_Batch_BP_OpenMP(model);

      model->loss_rprop = abs(model->get_loss(ENV::NONE));
      break;
    default:
      ERR(L"Optimizer type (%d | %ls) is not managed in the switch.",
          model->type_optimizer_function,
          OPTIMIZER_NAME[model->type_optimizer_function].c_str());
      break;
  }
}

void DatasetV1::Train_Epoch_Loop(Model *const model) {
  if (model->Use__Dropout__Bernoulli() ||
      model->Use__Dropout__Bernoulli__Inverted() ||
      model->Use__Dropout__Alpha()) {
    model->Dropout_Bernoulli();
  } else if (model->Use__Dropout__Zoneout()) {
    model->Dropout_Zoneout();
  }

  model->reset_loss();

  switch (model->type_optimizer_function) {
    case OPTIMIZER::GD:
    case OPTIMIZER::QUICKPROP:
    case OPTIMIZER::SARPROP:
    case OPTIMIZER::ADABOUND:
    case OPTIMIZER::ADAM:
    case OPTIMIZER::ADAMAX:
    case OPTIMIZER::AMSBOUND:
    case OPTIMIZER::AMSGRAD:
    case OPTIMIZER::NOSADAM:
      this->Train_Batch_BP_Loop(model);
      break;
    case OPTIMIZER::IRPROP_MINUS:
    case OPTIMIZER::IRPROP_PLUS:
      model->loss_rprop_tm1 = model->loss_rprop;

      this->Train_Batch_BP_Loop(model);

      model->loss_rprop = abs(model->get_loss(ENV::NONE));
      break;
    default:
      ERR(L"Optimizer type (%d | %ls) is not managed in the switch.",
          model->type_optimizer_function,
          OPTIMIZER_NAME[model->type_optimizer_function].c_str());
      break;
  }
}

void DatasetV1::Train_Batch_BP_OpenMP(Model *const model) {
  size_t const n_data(this->get_n_data()),
      tmp_maximum_batch_size(model->batch_size),
      tmp_number_batchs(static_cast<size_t>(
          ceil(static_cast<double>(n_data) /
               static_cast<double>(tmp_maximum_batch_size))));
  size_t tmp_batch_index(0_UZ), tmp_batch_size(0_UZ);

#pragma omp parallel private(tmp_batch_index, tmp_batch_size)
  for (tmp_batch_index = 0_UZ; tmp_batch_index != tmp_number_batchs;
       ++tmp_batch_index) {
    tmp_batch_size = tmp_batch_index + 1_UZ != tmp_number_batchs
                         ? tmp_maximum_batch_size
                         : n_data - tmp_batch_index * tmp_maximum_batch_size;

    model->forward_pass(
        tmp_batch_size,
        this->Get__Input_Array() + tmp_batch_index * tmp_maximum_batch_size);

    model->compute_error(
        tmp_batch_size,
        this->Get__Output_Array() + tmp_batch_index * tmp_maximum_batch_size);

    model->backward_pass(tmp_batch_size);

    model->update_derivatives(tmp_batch_size, model->ptr_array_layers + 1,
                                    model->ptr_last_layer);
  }

  model->merge_mp_accu_loss();

  model->n_acc_trial = n_data * (this->get_seq_w() - model->n_time_delay) *
                       (model->type_accuracy_function == ACCU_FN::CROSS_ENTROPY
                            ? 1_UZ
                            : model->get_n_out());
}

void DatasetV1::Train_Batch_BP_Loop(Model *const model) {
  size_t const n_data(this->get_n_data()),
      tmp_maximum_batch_size(model->batch_size),
      tmp_number_batchs(static_cast<size_t>(
          ceil(static_cast<double>(n_data) /
               static_cast<double>(tmp_maximum_batch_size))));
  size_t tmp_batch_size, tmp_batch_index;

  for (tmp_batch_index = 0_UZ; tmp_batch_index != tmp_number_batchs;
       ++tmp_batch_index) {
    tmp_batch_size = tmp_batch_index + 1_UZ != tmp_number_batchs
                         ? tmp_maximum_batch_size
                         : n_data - tmp_batch_index * tmp_maximum_batch_size;

    model->forward_pass(
        tmp_batch_size,
        this->Get__Input_Array() + tmp_batch_index * tmp_maximum_batch_size);

    model->compute_error(
        tmp_batch_size,
        this->Get__Output_Array() + tmp_batch_index * tmp_maximum_batch_size);

    model->backward_pass(tmp_batch_size);

    model->update_derivatives(tmp_batch_size, model->ptr_array_layers + 1,
                                    model->ptr_last_layer);
  }

  model->n_acc_trial = n_data * (this->get_seq_w() - model->n_time_delay) *
                       (model->type_accuracy_function == ACCU_FN::CROSS_ENTROPY
                            ? 1_UZ
                            : model->get_n_out());
}

bool DatasetV1::Initialize(void) {
  this->p_n_data_alloc = this->p_n_data = 0_UZ;
  this->p_n_inp = 0_UZ;
  this->p_n_out = 0_UZ;
  this->p_seq_w = 0_UZ;

  this->p_file_buffer_size = 32_UZ * KILOBYTE * KILOBYTE;  // byte(s).
  this->p_file_buffer_shift_size = 256_UZ * KILOBYTE;      // byte(s).

  this->Xm = nullptr;
  this->X = nullptr;

  this->Ym = nullptr;
  this->Y = nullptr;

  this->p_type_dataset_process = DATASET::BATCH;

  this->p_str_i = 0_UZ;

  return true;
}

bool DatasetV1::save(std::wstring const &path_name,
                   bool const normalize_received) {
  switch (this->p_type_data_file) {
    case DATASET_FORMAT::STANDARD:
      return (this->Save__Dataset(path_name + L".dataset", normalize_received));
      break;
    case DATASET_FORMAT::SPLIT:
      return (this->save_split_XY(path_name));
      break;
    default:
      ERR(L"DatasetV1 file type (%d | %ls) is not managed in the switch.",
          this->p_type_data_file,
          DATASET_FORMAT_NAME[this->p_type_data_file].c_str());
      return false;
  }

  return true;
}

bool DatasetV1::Save__Dataset(std::wstring const &path_name,
                            bool const normalize_received) {
  if (create_temp_file(path_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`create_temp_file(%ls)` function.",
        path_name.c_str());
    return false;
  }

  size_t const n_data(normalize_received ? this->DatasetV1::get_n_data()
                                         : this->p_n_data);
  size_t t, i, tmp_index;

  real const *const *const tmp_ptr_array_inputs_array(
      normalize_received ? this->DatasetV1::Get__Input_Array() : this->Xm),
      *const *const tmp_ptr_array_outputs_array(
          normalize_received ? this->DatasetV1::Get__Output_Array() : this->Ym);

  std::wstring tmp_string_write;

  std::wofstream file(CP_STR(path_name),
                      std::ios::out | std::ios::binary | std::ios::trunc);

  if (file.is_open() == false) {
    ERR(L"The file %ls can not be opened.", path_name.c_str());
    return false;
  }

  // Topology
  tmp_string_write =
      std::to_wstring(n_data) + L" " + std::to_wstring(this->p_n_inp) + L" " +
      std::to_wstring(this->p_n_out) + L" " + std::to_wstring(this->p_seq_w);

  // Input & Output
  for (i = 0_UZ; i != n_data; ++i) {
    for (t = 0_UZ; t != this->p_seq_w; ++t) {
      // Next line.
      tmp_string_write += CRLF;

      // Inputs [0...(N-1)]
      for (tmp_index = 0_UZ; tmp_index != this->p_n_inp - 1_UZ; ++tmp_index) {
        tmp_string_write +=
            to_wstring(
                tmp_ptr_array_inputs_array[i][t * this->p_n_inp + tmp_index],
                9u) +
            L" ";
      }

      // Last input
      tmp_string_write += to_wstring(
          tmp_ptr_array_inputs_array[i][t * this->p_n_inp + tmp_index], 9u);

      // Next line.
      tmp_string_write += CRLF;

      // Output [0...(N-1)]
      for (tmp_index = 0_UZ; tmp_index != this->p_n_out - 1_UZ; ++tmp_index) {
        tmp_string_write +=
            to_wstring(
                tmp_ptr_array_outputs_array[i][t * this->p_n_out + tmp_index],
                9u) +
            L" ";
      }

      // Last Output
      tmp_string_write += to_wstring(
          tmp_ptr_array_outputs_array[i][t * this->p_n_out + tmp_index], 9u);
    }

    if (tmp_string_write.size() >= this->p_file_buffer_size) {
      file.write(tmp_string_write.c_str(), static_cast<std::streamsize>(tmp_string_write.size()));
      tmp_string_write = L"";
    }
  }

  file.write(tmp_string_write.c_str(), static_cast<std::streamsize>(tmp_string_write.size()));

  file.flush();
  file.close();

  if (delete_temp_file(path_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`delete_temp_file(%ls)` function.",
        path_name.c_str());
    return false;
  }

  return true;
}

bool DatasetV1::Save__Dataset_Custom(std::wstring const &path_name) {
  if (create_temp_file(path_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`create_temp_file(%ls)` function.",
        path_name.c_str());
    return false;
  }

  size_t const n_data(this->p_n_data);
  size_t t, i, tmp_index;

  real const *const *const tmp_ptr_array_inputs_array(this->Xm),
      *const *const tmp_ptr_array_outputs_array(this->Ym);

  std::wstring tmp_string_write;

  std::wofstream file(CP_STR(path_name),
                      std::ios::out | std::ios::binary | std::ios::trunc);

  if (file.is_open() == false) {
    ERR(L"The file %ls can not be opened.", path_name.c_str());
    return false;
  }

  // Topology
  tmp_string_write = std::to_wstring(n_data) + L" " +
                     std::to_wstring(this->p_n_inp) + L" " +
                     std::to_wstring(this->p_n_out - 2_UZ) + L" " +
                     std::to_wstring(this->p_seq_w);

  // Input & Output
  for (i = 0_UZ; i != n_data; ++i) {
    for (t = 0_UZ; t != this->p_seq_w; ++t) {
      // Next line.
      tmp_string_write += CRLF;

      // Inputs [0...(N-1)]
      for (tmp_index = 0_UZ; tmp_index != this->p_n_inp - 1_UZ; ++tmp_index) {
        tmp_string_write +=
            to_wstring(
                tmp_ptr_array_inputs_array[i][t * this->p_n_inp + tmp_index],
                9u) +
            L" ";
      }

      // Last input
      tmp_string_write += to_wstring(
          tmp_ptr_array_inputs_array[i][t * this->p_n_inp + tmp_index], 9u);

      // Next line.
      tmp_string_write += CRLF;

      // Output [0...(N-1)]
      for (tmp_index = 2_UZ; tmp_index != this->p_n_out - 1_UZ; ++tmp_index) {
        if (tmp_index == 2_UZ) {
          tmp_string_write +=
              to_wstring(
                  tmp_ptr_array_outputs_array[i][t * this->p_n_out + 0_UZ] +
                      tmp_ptr_array_outputs_array[i][t * this->p_n_out + 1_UZ] +
                      tmp_ptr_array_outputs_array[i][t * this->p_n_out + 2_UZ],
                  9u) +
              L" ";
        } else {
          tmp_string_write +=
              to_wstring(
                  tmp_ptr_array_outputs_array[i][t * this->p_n_out + tmp_index],
                  9u) +
              L" ";
        }
      }

      // Last Output
      tmp_string_write += to_wstring(
          tmp_ptr_array_outputs_array[i][t * this->p_n_out + tmp_index], 9u);
    }

    if (tmp_string_write.size() >= this->p_file_buffer_size) {
      file.write(tmp_string_write.c_str(), static_cast<std::streamsize>(tmp_string_write.size()));
      tmp_string_write = L"";
    }
  }

  file.write(tmp_string_write.c_str(), static_cast<std::streamsize>(tmp_string_write.size()));

  file.flush();
  file.close();

  if (delete_temp_file(path_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`delete_temp_file(%ls)` function.",
        path_name.c_str());
    return false;
  }

  return true;
}

bool DatasetV1::save_split_XY(std::wstring const &path_name) {
  if (this->save_X(path_name + L".dataset-input") == false) {
    ERR(L"An error has been triggered from the "
        L"`save_X(%ls.dataset-input)` function.",
        path_name.c_str());
    return false;
  } else if (this->save_Y(path_name + L".dataset-output") == false) {
    ERR(L"An error has been triggered from the "
        L"`save_Y(%ls.dataset-output)` function.",
        path_name.c_str());
    return false;
  }

  return true;
}

bool DatasetV1::save_X(std::wstring const &path_name) {
  if (create_temp_file(path_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`create_temp_file(%ls)` function.",
        path_name.c_str());
    return false;
  }

  size_t t, i, tmp_index;

  std::wstring tmp_string_write;

  std::wofstream file(CP_STR(path_name),
                      std::ios::out | std::ios::binary | std::ios::trunc);

  if (file.is_open() == false) {
    ERR(L"The file %ls can not be opened.", path_name.c_str());
    return false;
  }

  // Topology
  tmp_string_write = std::to_wstring(this->p_n_data) + L" " +
                     std::to_wstring(this->p_n_inp) + L" " +
                     std::to_wstring(this->p_seq_w);

  for (i = 0_UZ; i != this->p_n_data; ++i) {
    for (t = 0_UZ; t != this->p_seq_w; ++t) {
      // Next line.
      tmp_string_write += CRLF;

      // Inputs [0...(N-1)]
      for (tmp_index = 0_UZ; tmp_index != this->p_n_inp - 1_UZ; ++tmp_index) {
        tmp_string_write +=
            to_wstring(
                this->Xm[i][t * this->p_n_inp + tmp_index], 9u) +
            L" ";
      }

      // Last input
      tmp_string_write += to_wstring(
          this->Xm[i][t * this->p_n_inp + tmp_index], 9u);
    }

    if (tmp_string_write.size() >= this->p_file_buffer_size) {
      file.write(tmp_string_write.c_str(), static_cast<std::streamsize>(tmp_string_write.size()));
      tmp_string_write = L"";
    }
  }

  file.write(tmp_string_write.c_str(), static_cast<std::streamsize>(tmp_string_write.size()));

  file.flush();
  file.close();

  if (delete_temp_file(path_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`delete_temp_file(%ls)` function.",
        path_name.c_str());
    return false;
  }

  return true;
}

bool DatasetV1::save_Y(std::wstring const &path_name) {
  if (create_temp_file(path_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`create_temp_file(%ls)` function.",
        path_name.c_str());
    return false;
  }

  size_t t, i, tmp_index;

  std::wstring tmp_string_write;

  std::wofstream file(CP_STR(path_name),
                      std::ios::out | std::ios::binary | std::ios::trunc);

  if (file.is_open() == false) {
    ERR(L"The file %ls can not be opened.", path_name.c_str());
    return false;
  }

  // Topology
  tmp_string_write = std::to_wstring(this->p_n_data) + L" " +
                     std::to_wstring(this->p_n_out) + L" " +
                     std::to_wstring(this->p_seq_w);

  for (i = 0_UZ; i != this->p_n_data; ++i) {
    for (t = 0_UZ; t != this->p_seq_w; ++t) {
      // Next line.
      tmp_string_write += CRLF;

      // Output [0...(N-1)]
      for (tmp_index = 0_UZ; tmp_index != this->p_n_out - 1_UZ; ++tmp_index) {
        tmp_string_write +=
            to_wstring(
                this->Ym[i][t * this->p_n_out + tmp_index], 9u) +
            L" ";
      }

      // Last Output
      tmp_string_write += to_wstring(
          this->Ym[i][t * this->p_n_out + tmp_index], 9u);
    }

    if (tmp_string_write.size() >= this->p_file_buffer_size) {
      file.write(tmp_string_write.c_str(), static_cast<std::streamsize>(tmp_string_write.size()));
      tmp_string_write = L"";
    }
  }

  file.write(tmp_string_write.c_str(), static_cast<std::streamsize>(tmp_string_write.size()));

  file.flush();
  file.close();

  if (delete_temp_file(path_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`delete_temp_file(%ls)` function.",
        path_name.c_str());
    return false;
  }

  return true;
}

bool DatasetV1::save(Model *const model, std::wstring path_name) {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");
    return false;
  } else if (this->valide_spec(model->n_inp, model->n_out, model->seq_w) ==
             false) {
    ERR(L"An error has been triggered from the "
        L"`valide_spec(%zu, %zu, %zu)` function.",
        model->n_inp, model->n_out, model->seq_w);
    return false;
  } else if (model->type != MODEL::AUTOENCODER) {
    ERR(L"The neural network (%ls) receive as argument need to be a %ls.",
        MODEL_NAME[model->type].c_str(),
        MODEL_NAME[MODEL::AUTOENCODER].c_str());
    return false;
  }

  auto tmp_Reset_IO_Mode(
      [tmp_use_first_layer_as_input = model->use_first_layer_as_input,
       tmp_use_last_layer_as_output = model->use_last_layer_as_output,
       &model]() -> bool {
        bool tmp_succes(true);

        if (model->Set__Input_Mode(tmp_use_first_layer_as_input) == false) {
          ERR(L"An error has been triggered from the "
              L"`Set__Input_Mode(%ls)` function.",
              to_wstring(tmp_use_first_layer_as_input).c_str());
          tmp_succes = false;
        }

        if (model->Set__Output_Mode(tmp_use_last_layer_as_output) == false) {
          ERR(L"An error has been triggered from the "
              L"`Set__Output_Mode(%ls)` function.",
              to_wstring(tmp_use_last_layer_as_output).c_str());
          tmp_succes = false;
        }

        return tmp_succes;
      });

  if (model->Set__Input_Mode(true) == false) {
    ERR(L"An error has been triggered from the "
        L"`Set__Input_Mode(true)` function.");
    return false;
  } else if (model->Set__Output_Mode(false) == false) {
    ERR(L"An error has been triggered from the "
        L"`Set__Output_Mode(false)` function.");
    if (tmp_Reset_IO_Mode() == false) {
      ERR(L"An error has been triggered from the "
          L"`tmp_Reset_IO_Mode()` function.");
      return false;
    }

    return false;
  } else if (model->update_mem_batch_size(this->p_n_data) == false) {
    ERR(L"An error has been triggered from the "
        L"`update_mem_batch_size(%zu)` function.",
        this->p_n_data);
    return false;
  }

  // By default save the dataset into .dataset extension.
  path_name += L".dataset";

  if (create_temp_file(path_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`create_temp_file(%ls)` function.",
        path_name.c_str());

    if (tmp_Reset_IO_Mode() == false) {
      ERR(L"An error has been triggered from the "
          L"`tmp_Reset_IO_Mode()` function.");
      return false;
    }

    return false;
  }

  size_t const n_data(this->p_n_data),
      tmp_maximum_batch_size(model->batch_size),
      tmp_number_batchs(static_cast<size_t>(
          ceil(static_cast<double>(n_data) /
               static_cast<double>(tmp_maximum_batch_size)))),
      tmp_number_outputs(model->get_n_out());
  size_t tmp_index, t, i, tmp_batch_size, tmp_batch_index;

  var const *Q;

  std::wstring tmp_string_write;

  std::wofstream file(CP_STR(path_name),
                      std::ios::out | std::ios::binary | std::ios::trunc);

  if (file.is_open() == false) {
    ERR(L"The file %ls can not be opened.", path_name.c_str());
    return false;
  }

  // Topology
  tmp_string_write = std::to_wstring(n_data) + L" " +
                     std::to_wstring(tmp_number_outputs) + L" " +
                     std::to_wstring(tmp_number_outputs) + L" " +
                     std::to_wstring(this->p_seq_w);

  // Input & Output
  for (tmp_batch_index = 0_UZ; tmp_batch_index != tmp_number_batchs;
       ++tmp_batch_index) {
    tmp_batch_size = tmp_batch_index + 1_UZ != tmp_number_batchs
                         ? tmp_maximum_batch_size
                         : n_data - tmp_batch_index * tmp_maximum_batch_size;

    model->forward_pass(tmp_batch_size,
                        this->Xm + tmp_batch_index * tmp_maximum_batch_size);

    for (i = 0_UZ; i != tmp_batch_size; ++i) {
      for (t = 0_UZ; t != this->p_seq_w; ++t) {
        Q = model->get_out(i, t);

        // Input
        tmp_string_write += CRLF;
        for (tmp_index = 0_UZ; tmp_index != tmp_number_outputs; ++tmp_index) {
          tmp_string_write +=
              to_wstring(cast(Q[tmp_index]), 9);

          if (tmp_index + 1_UZ != tmp_number_outputs) tmp_string_write += L" ";
        }

        // Output
        tmp_string_write += CRLF;
        for (tmp_index = 0_UZ; tmp_index != tmp_number_outputs; ++tmp_index) {
          tmp_string_write +=
              to_wstring(cast(Q[tmp_index]), 9);

          if (tmp_index + 1_UZ != tmp_number_outputs) tmp_string_write += L" ";
        }
      }
    }
  }

  file.write(tmp_string_write.c_str(), static_cast<std::streamsize>(tmp_string_write.size()));

  file.flush();
  file.close();

  if (delete_temp_file(path_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`delete_temp_file(%ls)` function.",
        path_name.c_str());

    if (tmp_Reset_IO_Mode() == false) {
      ERR(L"An error has been triggered from the "
          L"`tmp_Reset_IO_Mode()` function.");
      return false;
    }

    return false;
  } else if (tmp_Reset_IO_Mode() == false) {
    ERR(L"An error has been triggered from the "
        L"`tmp_Reset_IO_Mode()` function.");
    return false;
  }

  return true;
}

bool DatasetV1::shift(size_t const shift, DATA::TYPE const data_type) {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");
    return false;
  } else if (shift == 0_UZ) {
    ERR(L"shift can not be zero.");
    return false;
  } else if (shift >= this->p_n_data) {
    ERR(L"shift (%zu) can not be greater or equal to the number of data (%zu).",
        shift, this->p_n_data);
    return false;
  } else if (this->_reference) {
    ERR(L"The dataset is allocate as refence.");
    return false;
  }

  size_t const n_data_new(this->p_n_data - shift);
  size_t i;

  if (data_type == DATA::INPUT) {
    for (i = 0_UZ; i != n_data_new; ++i)
      memcpy(this->X + i * this->p_n_inp * this->p_seq_w,
             this->X + (i + 1_UZ) * this->p_n_inp * this->p_seq_w,
             this->p_n_inp * this->p_seq_w * sizeof(real));
  } else if (data_type == DATA::OUTPUT) {
    for (i = 0_UZ; i != n_data_new; ++i)
      memcpy(this->Y + i * this->p_n_out * this->p_seq_w,
             this->Y + (i + 1_UZ) * this->p_n_out * this->p_seq_w,
             this->p_n_out * this->p_seq_w * sizeof(real));
  } else {
    ERR(L"Type input (%d) is not managed in the function.", data_type);
    return false;
  }

  // Inputs.
  real *X(Mem::reallocate(this->X, this->p_n_inp * n_data_new * this->p_seq_w,
                          this->p_n_inp * this->p_n_data * this->p_seq_w));

  this->X = X;
  // |END| Inputs. |END|

  // Outputs.
  real *Y(Mem::reallocate(this->Y, this->p_n_out * n_data_new * this->p_seq_w,
                          this->p_n_out * this->p_n_data * this->p_seq_w));

  this->Y = Y;
  // |END| Outputs. |END|

  this->p_n_data = n_data_new;

  for (i = 0_UZ; i != this->p_n_data; ++i) {
    this->Xm[i] = X + i * this->p_n_inp * this->p_seq_w;
    this->Ym[i] = Y + i * this->p_n_out * this->p_seq_w;
  }

  return true;
}

bool DatasetV1::Time_Direction(real const minimum_range_received,
                             real const maximum_range_received,
                             DATA::TYPE const data_type) {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");
    return false;
  } else if (minimum_range_received > maximum_range_received) {
    ERR(L"Minimum range (%f) can not be greater than maximum range (%f).",
        minimum_range_received, maximum_range_received);
    return false;
  } else if (this->p_seq_w <= 1_UZ) {
    ERR(L"Recurrent depth can not be less or equal to one.");
    return false;
  } else if (this->_reference) {
    ERR(L"The dataset is allocate as refence.");
    return false;
  }

  size_t const n_data(this->p_n_data), tmp_input_size(this->p_n_inp),
      tmp_number_outputs(this->p_n_out), seq_w(this->p_seq_w);
  size_t i, t, k;

  real tmp_direction;

  if (data_type == DATA::INPUT) {
    for (i = 0_UZ; i != n_data; ++i) {
      // Time step real...1.
      for (t = seq_w; --t > 0;) {
        // Input N...1.
        for (k = tmp_input_size; --k > 0;) {
          tmp_direction =
              Math::sign(this->Xm[i][t * tmp_input_size + k] -
                         this->Xm[i][t * tmp_input_size + k - 1_UZ]);

          switch (static_cast<int>(tmp_direction)) {
            case 1:
              tmp_direction = maximum_range_received;
              break;
            case -1:
              tmp_direction = minimum_range_received;
              break;
            case 0:
              break;
            default:
              ERR(L"Direction (%f) is not managed in the switch.",
                  tmp_direction);
              break;
          }

          this->X[i * tmp_input_size * seq_w + t * tmp_input_size + k] =
              tmp_direction;
        }

        // First input.
        tmp_direction = Math::sign(this->Xm[i][t * tmp_input_size] -
                                   this->Xm[i][(t - 1_UZ) * tmp_input_size]);

        switch (static_cast<int>(tmp_direction)) {
          case 1:
            tmp_direction = maximum_range_received;
            break;
          case -1:
            tmp_direction = minimum_range_received;
            break;
          case 0:
            break;
          default:
            ERR(L"Direction (%f) is not managed in the switch.",
                tmp_direction);
            break;
        }

        this->X[i * tmp_input_size * seq_w + t * tmp_input_size] =
            tmp_direction;
      }

      // First time step.
      //  Input N...1.
      for (k = tmp_input_size; --k > 0;) {
        tmp_direction = Math::sign(this->Xm[i][k] - this->Xm[i][k - 1_UZ]);

        switch (static_cast<int>(tmp_direction)) {
          case 1:
            tmp_direction = maximum_range_received;
            break;
          case -1:
            tmp_direction = minimum_range_received;
            break;
          case 0:
            break;
          default:
            ERR(L"Direction (%f) is not managed in the switch.",
                tmp_direction);
            break;
        }

        this->X[i * tmp_input_size * seq_w + k] = tmp_direction;
      }

      //  First input.
      this->X[i * tmp_input_size * seq_w] = 0_r;
      // |END| First time step. |END|
    }
  } else if (data_type == DATA::OUTPUT) {
    for (i = 0_UZ; i != n_data; ++i) {
      // Time step real...1.
      for (t = seq_w; --t > 0;) {
        // Input N...1.
        for (k = tmp_number_outputs; --k > 0;) {
          tmp_direction =
              Math::sign(this->Ym[i][t * tmp_number_outputs + k] -
                         this->Ym[i][t * tmp_number_outputs + k - 1_UZ]);

          switch (static_cast<int>(tmp_direction)) {
            case 1:
              tmp_direction = maximum_range_received;
              break;
            case -1:
              tmp_direction = minimum_range_received;
              break;
            case 0:
              break;
            default:
              ERR(L"Direction (%f) is not managed in the switch.",
                  tmp_direction);
              break;
          }

          this->Y[i * tmp_number_outputs * seq_w + t * tmp_number_outputs + k] =
              tmp_direction;
        }

        // First input.
        tmp_direction =
            Math::sign(this->Ym[i][t * tmp_number_outputs] -
                       this->Ym[i][(t - 1_UZ) * tmp_number_outputs]);

        switch (static_cast<int>(tmp_direction)) {
          case 1:
            tmp_direction = maximum_range_received;
            break;
          case -1:
            tmp_direction = minimum_range_received;
            break;
          case 0:
            break;
          default:
            ERR(L"Direction (%f) is not managed in the switch.",
                tmp_direction);
            break;
        }

        this->Y[i * tmp_number_outputs * seq_w + t * tmp_number_outputs] =
            tmp_direction;
      }

      // First time step.
      //  Input N...1.
      for (k = tmp_number_outputs; --k > 0;) {
        tmp_direction = Math::sign(this->Ym[i][k] - this->Ym[i][k - 1_UZ]);

        switch (static_cast<int>(tmp_direction)) {
          case 1:
            tmp_direction = maximum_range_received;
            break;
          case -1:
            tmp_direction = minimum_range_received;
            break;
          case 0:
            break;
          default:
            ERR(L"Direction (%f) is not managed in the switch.",
                tmp_direction);
            break;
        }

        this->Y[i * tmp_number_outputs * seq_w + k] = tmp_direction;
      }

      //  First input.
      this->Y[i * tmp_number_outputs * seq_w] = 0_r;
      // |END| First time step. |END|
    }
  } else {
    ERR(L"Type input (%f) is not managed in the function.", data_type);
    return false;
  }

  return true;
}

bool DatasetV1::Input_To_Output(DATA::TYPE const data_type) {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");
    return false;
  } else if (this->_reference) {
    ERR(L"The dataset is allocate as refence.");
    return false;
  }

  if (data_type == DATA::INPUT) {
    size_t const tmp_input_size(this->p_n_out);
    size_t t, i, tmp_index;

    real *tmp_ptr_array_outputs(Mem::reallocate<real, false>(
        this->X, tmp_input_size * this->p_n_data * this->p_seq_w,
        this->p_n_inp * this->p_n_data * this->p_seq_w));

    if (tmp_ptr_array_outputs == nullptr) {
      ERR(L"An error has been triggered from the "
          L"`reallocate_cpp<%zu>(ptr, %zu, %zu, false)` function.",
          sizeof(real), tmp_input_size * this->p_n_data * this->p_seq_w,
          this->p_n_inp * this->p_n_data * this->p_seq_w);
      return false;
    }

    this->X = tmp_ptr_array_outputs;

    for (i = 0_UZ; i != this->p_n_data; ++i)
      this->Xm[i] = tmp_ptr_array_outputs + i * tmp_input_size * this->p_seq_w;

    for (i = 0_UZ; i != this->p_n_data; ++i)
      for (t = 0_UZ; t != this->p_seq_w; ++t)
        for (tmp_index = 0_UZ; tmp_index != tmp_input_size; ++tmp_index)
          *tmp_ptr_array_outputs++ =
              this->Ym[i][t * tmp_input_size + tmp_index];

    this->p_n_inp = tmp_input_size;
  } else if (data_type == DATA::OUTPUT) {
    size_t const tmp_input_size(this->p_n_inp);
    size_t t, i, tmp_index;

    real *tmp_ptr_array_outputs(Mem::reallocate<real, false>(
        this->Y, tmp_input_size * this->p_n_data * this->p_seq_w,
        this->p_n_out * this->p_n_data * this->p_seq_w));

    if (tmp_ptr_array_outputs == nullptr) {
      ERR(L"An error has been triggered from the "
          L"`reallocate_cpp<%zu>(ptr, %zu, %zu, false)` function.",
          sizeof(real), tmp_input_size * this->p_n_data * this->p_seq_w,
          this->p_n_out * this->p_n_data * this->p_seq_w);
      return false;
    }

    this->Y = tmp_ptr_array_outputs;

    for (i = 0_UZ; i != this->p_n_data; ++i)
      this->Ym[i] = tmp_ptr_array_outputs + i * tmp_input_size * this->p_seq_w;

    for (i = 0_UZ; i != this->p_n_data; ++i)
      for (t = 0_UZ; t != this->p_seq_w; ++t)
        for (tmp_index = 0_UZ; tmp_index != tmp_input_size; ++tmp_index)
          *tmp_ptr_array_outputs++ =
              this->Xm[i][t * tmp_input_size + tmp_index];

    this->p_n_out = tmp_input_size;
  } else {
    ERR(L"Type input (%d) is not managed in the function.", data_type);
    return false;
  }

  return true;
}

bool DatasetV1::Unrecurrent(void) {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");
    return false;
  } else if (this->_reference) {
    ERR(L"The dataset is allocate as refence.");
    return false;
  } else if (this->p_seq_w <= 1_UZ) {
    return true;
  }

  size_t const tmp_last_time_step_index(this->p_seq_w - 1_UZ);
  size_t i, tmp_index;

  real *tmp_ptr_array_inputs(new real[this->p_n_data * this->p_n_inp]),
      *tmp_ptr_array_outputs(new real[this->p_n_data * this->p_n_out]);

  for (i = 0_UZ; i != this->p_n_data; ++i) {
    for (tmp_index = 0_UZ; tmp_index != this->p_n_inp; ++tmp_index)
      tmp_ptr_array_inputs[i * this->p_n_inp + tmp_index] =
          this->Xm[i][tmp_last_time_step_index * this->p_n_inp + tmp_index];

    for (tmp_index = 0_UZ; tmp_index != this->p_n_out; ++tmp_index)
      tmp_ptr_array_outputs[i * this->p_n_out + tmp_index] =
          this->Ym[i][tmp_last_time_step_index * this->p_n_out + tmp_index];
  }

  delete[](this->X);
  this->X = tmp_ptr_array_inputs;

  delete[](this->Y);
  this->Y = tmp_ptr_array_outputs;

  for (i = 0_UZ; i != this->p_n_data; ++i) {
    this->Xm[i] = tmp_ptr_array_inputs + i * this->p_n_inp;
    this->Ym[i] = tmp_ptr_array_outputs + i * this->p_n_out;
  }

  this->p_seq_w = 1_UZ;

  return true;
}

bool DatasetV1::Remove(size_t const input_index_received,
                     DATA::TYPE const data_type) {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");
    return false;
  } else if (this->_reference) {
    ERR(L"The dataset is allocate as refence.");
    return false;
  }

  if (data_type == DATA::INPUT) {
    if (input_index_received >= this->p_n_inp) {
      ERR(L"Input index (%zu) overflow (%zu).", input_index_received,
          this->p_n_inp);
      return false;
    }

    size_t const tmp_new_input_size(this->p_n_inp - 1_UZ);
    size_t i, t, tmp_shifted_index, tmp_index;

    real *tmp_ptr_array_inputs(
        new real[this->p_n_data * this->p_seq_w * tmp_new_input_size]);

    for (i = 0_UZ; i != this->p_n_data; ++i) {
      for (t = 0_UZ; t != this->p_seq_w; ++t) {
        // Left.
        for (tmp_index = 0_UZ; tmp_index != input_index_received; ++tmp_index)
          tmp_ptr_array_inputs[i * tmp_new_input_size * this->p_seq_w +
                               t * tmp_new_input_size + tmp_index] =
              this->Xm[i][t * this->p_n_inp + tmp_index];

        // Right.
        for (tmp_shifted_index = input_index_received,
            tmp_index = input_index_received + 1_UZ;
             tmp_index != this->p_n_inp; ++tmp_index)
          tmp_ptr_array_inputs[i * tmp_new_input_size * this->p_seq_w +
                               t * tmp_new_input_size + tmp_shifted_index] =
              this->Xm[i][t * this->p_n_inp + tmp_index];
      }
    }

    delete[](this->X);
    this->X = tmp_ptr_array_inputs;

    for (i = 0_UZ; i != this->p_n_data; ++i)
      this->Xm[i] =
          tmp_ptr_array_inputs + i * tmp_new_input_size * this->p_seq_w;

    --this->p_n_inp;
  } else if (data_type == DATA::OUTPUT) {
    if (input_index_received >= this->p_n_out) {
      ERR(L"Input index (%zu) overflow (%zu).", input_index_received,
          this->p_n_out);
      return false;
    }

    size_t const tmp_new_input_size(this->p_n_out - 1_UZ);
    size_t i, t, tmp_shifted_index, tmp_index;

    real *tmp_ptr_array_outputs(
        new real[this->p_n_data * this->p_seq_w * tmp_new_input_size]);

    for (i = 0_UZ; i != this->p_n_data; ++i) {
      for (t = 0_UZ; t != this->p_seq_w; ++t) {
        // Left.
        for (tmp_index = 0_UZ; tmp_index != input_index_received; ++tmp_index)
          tmp_ptr_array_outputs[i * tmp_new_input_size * this->p_seq_w +
                                t * tmp_new_input_size + tmp_index] =
              this->Ym[i][t * this->p_n_out + tmp_index];

        // Right.
        for (tmp_shifted_index = input_index_received,
            tmp_index = input_index_received + 1_UZ;
             tmp_index != this->p_n_out; ++tmp_index)
          tmp_ptr_array_outputs[i * tmp_new_input_size * this->p_seq_w +
                                t * tmp_new_input_size + tmp_shifted_index] =
              this->Ym[i][t * this->p_n_out + tmp_index];
      }
    }

    delete[](this->Y);
    this->Y = tmp_ptr_array_outputs;

    for (i = 0_UZ; i != this->p_n_data; ++i)
      this->Ym[i] =
          tmp_ptr_array_outputs + i * tmp_new_input_size * this->p_seq_w;

    --this->p_n_out;
  } else {
    ERR(L"Type input (%d) is not managed in the function.", data_type);
    return false;
  }

  return true;
}

bool DatasetV1::allocate(DATASET_FORMAT::TYPE const dset_fmt,
                       std::wstring const &path_name) {
  switch (dset_fmt) {
    case DATASET_FORMAT::STANDARD:
      if (this->Allocate__Dataset(path_name + L".dataset") == false) {
        ERR(L"An error has been triggered from the "
            L"`Allocate__Dataset(%ls.dataset)` function.",
            path_name.c_str());
        return false;
      }
      break;
    case DATASET_FORMAT::SPLIT:
      if (this->Allocate__Dataset_Split(path_name) == false) {
        ERR(L"An error has been triggered from the "
            L"`Allocate__Dataset_Split(%ls)` function.",
            path_name.c_str());
        return false;
      }
      break;
    case DATASET_FORMAT::MNIST:
      if (this->Allocate__MNIST(path_name) == false) {
        ERR(L"An error has been triggered from the "
            L"`Allocate__MNIST(%ls)` function.",
            path_name.c_str());
        return false;
      }
      break;
    default:
      ERR(L"DatasetV1 file type (%d | %ls) is not managed in the switch.",
          dset_fmt, DATASET_FORMAT_NAME[dset_fmt].c_str());
      return false;
  }

  this->p_type_data_file = dset_fmt;

  this->Check_Use__Label();

  this->_reference = false;

  return true;
}

void DatasetV1::Check_Use__Label(void) {
  if (this->get_n_out() == 1_UZ) {
    this->_use_multi_label = false;
  } else {
    size_t tmp_numbre_labels(0_UZ);

    for (size_t k(0_UZ); k != this->p_n_out; ++k)
      if (this->Ym[0][k] != 0_r)
        ++tmp_numbre_labels;

    if (tmp_numbre_labels > 1_UZ)
      this->_use_multi_label = true;
  }
}

void DatasetV1::Compute__Start_Index(void) {
  size_t tmp_start_index(0_UZ);

  if (this->Xm_coeff_size != nullptr) {
    size_t tmp_J_level, tmp_j_index, k, tmp_circularity_end;

    for (k = 0_UZ; k != this->p_n_inp; ++k) {
      tmp_J_level = this->Xm_coeff_size[k] / this->p_n_data;

      if (tmp_J_level > 1_UZ) {
        --tmp_J_level;

        tmp_circularity_end = 1_UZ;

        for (tmp_j_index = 1_UZ; tmp_j_index != tmp_J_level; ++tmp_j_index)
          tmp_circularity_end = 2_UZ * tmp_circularity_end + 1_UZ;

        tmp_start_index =
            std::max<size_t>(tmp_start_index, tmp_circularity_end);
      }
    }
  }

  if (this->Ym_coeff_size != nullptr) {
    size_t tmp_J_level, tmp_j_index, k, tmp_circularity_end;

    for (k = 0_UZ; k != this->p_n_out; ++k) {
      tmp_J_level = this->Ym_coeff_size[k] / this->p_n_data;

      if (tmp_J_level > 1_UZ) {
        --tmp_J_level;

        tmp_circularity_end = 1_UZ;

        for (tmp_j_index = 1_UZ; tmp_j_index != tmp_J_level; ++tmp_j_index)
          tmp_circularity_end = 2_UZ * tmp_circularity_end + 1_UZ;

        tmp_start_index =
            std::max<size_t>(tmp_start_index, tmp_circularity_end);
      }
    }
  }

  this->p_str_i = tmp_start_index;
}

bool DatasetV1::Set__Type_Data_File(
    DATASET_FORMAT::TYPE const type_dataset_file_received) {
  if (type_dataset_file_received >= DATASET_FORMAT::LENGTH) {
    ERR(L"DatasetV1 file type (%d | %ls) is not managed in the switch.",
        type_dataset_file_received,
        DATASET_FORMAT_NAME[type_dataset_file_received].c_str());
    return false;
  }

  this->p_type_data_file = type_dataset_file_received;

  return true;
}

bool DatasetV1::Allocate__Dataset(std::wstring const &path_name) {
  wchar_t *tmp_ptr_array_buffers(nullptr), *tmp_ptr_last_buffer(nullptr);

  size_t tmp_block_size, n_data, tmp_input_size, tmp_number_outputs, seq_w,
      tmp_index, i, t;

  real const **tmp_ptr_array_inputs_array(nullptr),
      **tmp_ptr_array_outputs_array(nullptr);
  real *tmp_ptr_array_inputs, *tmp_ptr_array_outputs;

  double tmp_output;

  std::vector<wchar_t> tmp_vector_buffers;

  std::wifstream file(CP_STR(path_name),
                      std::ios::in | std::ios::binary);

  if (file.is_open() == false) {
    ERR(L"The file %ls can not be opened.", path_name.c_str());
    return false;
  }

  if (file.eof()) {
    ERR(L"File `%ls` is empty.", path_name.c_str());
    return false;
  }

  if (read_stream_block_n_parse<size_t>(
          tmp_ptr_array_buffers, tmp_ptr_last_buffer, tmp_block_size,
          this->p_file_buffer_size, this->p_file_buffer_shift_size, n_data,
          tmp_vector_buffers, file, L'\n') == false) {
    ERR(L"An error has been triggered from the "
        L"`read_stream_block_n_parse(ptr, ptr, %zu, %zu, %zu, ptr, vector, "
        L"wifstream, '\\n')` function, while reading the number of examples.",
        tmp_block_size, this->p_file_buffer_size,
        this->p_file_buffer_shift_size);
    return false;
  } else if (read_stream_block_n_parse<size_t>(
                 tmp_ptr_array_buffers, tmp_ptr_last_buffer, tmp_block_size,
                 this->p_file_buffer_size, this->p_file_buffer_shift_size,
                 tmp_input_size, tmp_vector_buffers, file, L'\n') == false) {
    ERR(L"An error has been triggered from the "
        L"`read_stream_block_n_parse(ptr, ptr, %zu, %zu, %zu, ptr, vector, "
        L"wifstream, '\\n')` function, while reading the number of inputs.",
        tmp_block_size, this->p_file_buffer_size,
        this->p_file_buffer_shift_size);
    return false;
  } else if (read_stream_block_n_parse<size_t>(
                 tmp_ptr_array_buffers, tmp_ptr_last_buffer, tmp_block_size,
                 this->p_file_buffer_size, this->p_file_buffer_shift_size,
                 tmp_number_outputs, tmp_vector_buffers, file,
                 L'\n') == false) {
    ERR(L"An error has been triggered from the "
        L"`read_stream_block_n_parse(ptr, ptr, %zu, %zu, %zu, ptr, vector, "
        L"wifstream, '\\n')` function, while reading the number of outputs.",
        tmp_block_size, this->p_file_buffer_size,
        this->p_file_buffer_shift_size);
    return false;
  } else if (read_stream_block_n_parse<size_t>(
                 tmp_ptr_array_buffers, tmp_ptr_last_buffer, tmp_block_size,
                 this->p_file_buffer_size, this->p_file_buffer_shift_size,
                 seq_w, tmp_vector_buffers, file, L'\n') == false) {
    ERR(L"An error has been triggered from the "
        L"`read_stream_block_n_parse(ptr, ptr, %zu, %zu, %zu, ptr, vector, "
        L"wifstream, '\\n')` function, while reading the number of recurrent "
        L"depth.",
        tmp_block_size, this->p_file_buffer_size,
        this->p_file_buffer_shift_size);
    return false;
  }

  if (file.fail()) {
    ERR(L"Logical error on i/o operation \"%ls\".", path_name.c_str());
    return false;
  }

  tmp_ptr_array_inputs_array = new real const *[n_data];
  tmp_ptr_array_outputs_array = new real const *[n_data];
  tmp_ptr_array_inputs = new real[tmp_input_size * n_data * seq_w];
  tmp_ptr_array_outputs = new real[tmp_number_outputs * n_data * seq_w];

  for (i = 0_UZ; i != n_data; ++i) {
    tmp_ptr_array_inputs_array[i] =
        tmp_ptr_array_inputs + i * tmp_input_size * seq_w;

    tmp_ptr_array_outputs_array[i] =
        tmp_ptr_array_outputs + i * tmp_number_outputs * seq_w;
  }

  for (i = 0_UZ; i != n_data; ++i) {
    for (t = 0_UZ; t != seq_w; ++t) {
      for (tmp_index = 0_UZ; tmp_index != tmp_input_size;
           ++tmp_index, ++tmp_ptr_array_inputs) {
        if (read_stream_block_n_parse<double>(
                tmp_ptr_array_buffers, tmp_ptr_last_buffer, tmp_block_size,
                this->p_file_buffer_size, this->p_file_buffer_shift_size,
                tmp_output, tmp_vector_buffers, file, L'\n') == false) {
          ERR(L"An error has been triggered from the "
              L"`read_stream_block_n_parse(ptr, ptr, %zu, %zu, %zu, ptr, "
              L"vector, wifstream, '\\n')` function, while reading data %zu "
              L"at %zu input.",
              tmp_block_size, this->p_file_buffer_size,
              this->p_file_buffer_shift_size, i, tmp_index);

          delete[](tmp_ptr_array_inputs_array[0]);
          delete[](tmp_ptr_array_inputs_array);
          delete[](tmp_ptr_array_outputs_array[0]);
          delete[](tmp_ptr_array_outputs_array);

          return false;
        }

        *tmp_ptr_array_inputs = static_cast<real>(tmp_output);
      }

      for (tmp_index = 0_UZ; tmp_index != tmp_number_outputs;
           ++tmp_index, ++tmp_ptr_array_outputs) {
        if (read_stream_block_n_parse<double>(
                tmp_ptr_array_buffers, tmp_ptr_last_buffer, tmp_block_size,
                this->p_file_buffer_size, this->p_file_buffer_shift_size,
                tmp_output, tmp_vector_buffers, file, L'\n') == false) {
          ERR(L"An error has been triggered from the "
              L"`read_stream_block_n_parse(ptr, ptr, %zu, %zu, %zu, ptr, "
              L"vector, wifstream, '\\n')` function, while reading data %zu "
              L"at %zu output.",
              tmp_block_size, this->p_file_buffer_size,
              this->p_file_buffer_shift_size, i, tmp_index);

          delete[](tmp_ptr_array_inputs_array[0]);
          delete[](tmp_ptr_array_inputs_array);
          delete[](tmp_ptr_array_outputs_array[0]);
          delete[](tmp_ptr_array_outputs_array);

          return false;
        }

        *tmp_ptr_array_outputs = static_cast<real>(tmp_output);
      }
    }
  }

  if (file.fail()) {
    ERR(L"Logical error on i/o operation \"%ls\".", path_name.c_str());
    return false;
  }

  file.close();

  this->p_n_data_alloc = this->p_n_data = n_data;
  this->p_n_inp = tmp_input_size;
  this->p_n_out = tmp_number_outputs;
  this->p_seq_w = seq_w;

  this->Xm = tmp_ptr_array_inputs_array;
  this->X = tmp_ptr_array_inputs - n_data * tmp_input_size * seq_w;
  this->Ym = tmp_ptr_array_outputs_array;
  this->Y = tmp_ptr_array_outputs - n_data * tmp_number_outputs * seq_w;

  return true;
}

bool DatasetV1::Allocate__Dataset_Split(std::wstring const &path_name) {
  if (this->Allocate__Dataset_Split__Input(path_name + L".dataset-input") ==
      false) {
    ERR(L"An error has been triggered from the "
        L"`Allocate__Dataset_Split__Input(%ls.dataset-input)` function.",
        path_name.c_str());
    return false;
  } else if (this->Allocate__Dataset_Split__Output(
                 path_name + L".dataset-output") == false) {
    ERR(L"An error has been triggered from the "
        L"`Allocate__Dataset_Split__Output(%ls.dataset-output)` function.",
        path_name.c_str());
    return false;
  }

  return true;
}

bool DatasetV1::Allocate__Dataset_Split__Input(std::wstring const &path_name) {
  wchar_t *tmp_ptr_array_buffers(nullptr), *tmp_ptr_last_buffer(nullptr);

  size_t tmp_block_size, n_data, tmp_input_size, seq_w, tmp_index, i, t;

  real const **tmp_ptr_array_inputs_array(nullptr);
  real *tmp_ptr_array_inputs;

  double tmp_output;

  std::vector<wchar_t> tmp_vector_buffers;

  std::wifstream file(CP_STR(path_name),
                      std::ios::in | std::ios::binary);

  if (file.is_open() == false) {
    ERR(L"The file %ls can not be opened.", path_name.c_str());
    return false;
  }

  if (file.eof()) {
    ERR(L"File `%ls` is empty.", path_name.c_str());
    return false;
  }

  if (read_stream_block_n_parse<size_t>(
          tmp_ptr_array_buffers, tmp_ptr_last_buffer, tmp_block_size,
          this->p_file_buffer_size, this->p_file_buffer_shift_size, n_data,
          tmp_vector_buffers, file, L'\n') == false) {
    ERR(L"An error has been triggered from the "
        L"`read_stream_block_n_parse(ptr, ptr, %zu, %zu, %zu, ptr, vector, "
        L"wifstream, '\\n')` function, while reading the number of examples.",
        tmp_block_size, this->p_file_buffer_size,
        this->p_file_buffer_shift_size);
    return false;
  } else if (read_stream_block_n_parse<size_t>(
                 tmp_ptr_array_buffers, tmp_ptr_last_buffer, tmp_block_size,
                 this->p_file_buffer_size, this->p_file_buffer_shift_size,
                 tmp_input_size, tmp_vector_buffers, file, L'\n') == false) {
    ERR(L"An error has been triggered from the "
        L"`read_stream_block_n_parse(ptr, ptr, %zu, %zu, %zu, ptr, vector, "
        L"wifstream, '\\n')` function, while reading the number of inputs.",
        tmp_block_size, this->p_file_buffer_size,
        this->p_file_buffer_shift_size);
    return false;
  } else if (read_stream_block_n_parse<size_t>(
                 tmp_ptr_array_buffers, tmp_ptr_last_buffer, tmp_block_size,
                 this->p_file_buffer_size, this->p_file_buffer_shift_size,
                 seq_w, tmp_vector_buffers, file, L'\n') == false) {
    ERR(L"An error has been triggered from the "
        L"`read_stream_block_n_parse(ptr, ptr, %zu, %zu, %zu, ptr, vector, "
        L"wifstream, '\\n')` function, while reading the number of recurrent "
        L"depth.",
        tmp_block_size, this->p_file_buffer_size,
        this->p_file_buffer_shift_size);
    return false;
  }

  tmp_ptr_array_inputs_array = new real const *[n_data];
  tmp_ptr_array_inputs = new real[tmp_input_size * n_data * seq_w];

  for (i = 0_UZ; i != n_data; ++i) {
    tmp_ptr_array_inputs_array[i] =
        tmp_ptr_array_inputs + i * tmp_input_size * seq_w;
  }

  for (i = 0_UZ; i != n_data; ++i) {
    for (t = 0_UZ; t != seq_w; ++t) {
      for (tmp_index = 0_UZ; tmp_index != tmp_input_size;
           ++tmp_index, ++tmp_ptr_array_inputs) {
        if (read_stream_block_n_parse<double>(
                tmp_ptr_array_buffers, tmp_ptr_last_buffer, tmp_block_size,
                this->p_file_buffer_size, this->p_file_buffer_shift_size,
                tmp_output, tmp_vector_buffers, file, L'\n') == false) {
          ERR(L"An error has been triggered from the "
              L"`read_stream_block_n_parse(ptr, ptr, %zu, %zu, %zu, ptr, "
              L"vector, wifstream, '\\n')` function, while reading data %zu "
              L"at %zu input.",
              tmp_block_size, this->p_file_buffer_size,
              this->p_file_buffer_shift_size, i, tmp_index);

          delete[](tmp_ptr_array_inputs_array[0]);
          delete[](tmp_ptr_array_inputs_array);

          return false;
        }

        *tmp_ptr_array_inputs = static_cast<real>(tmp_output);
      }
    }
  }

  if (file.fail()) {
    ERR(L"Logical error on i/o operation \"%ls\".", path_name.c_str());
    return false;
  }

  file.close();

  this->p_n_data_alloc = this->p_n_data = n_data;
  this->p_n_inp = tmp_input_size;
  this->p_seq_w = seq_w;

  this->Xm = tmp_ptr_array_inputs_array;
  this->X = tmp_ptr_array_inputs - n_data * tmp_input_size * seq_w;

  return true;
}

bool DatasetV1::Allocate__Dataset_Split__Output(std::wstring const &path_name) {
  wchar_t *tmp_ptr_array_buffers(nullptr), *tmp_ptr_last_buffer(nullptr);

  size_t tmp_block_size, n_data, tmp_number_outputs, seq_w, tmp_index, i, t;

  real const **tmp_ptr_array_outputs_array(nullptr);
  real *tmp_ptr_array_outputs;

  double tmp_output;

  std::vector<wchar_t> tmp_vector_buffers;

  std::wifstream file(CP_STR(path_name),
                      std::ios::in | std::ios::binary);

  if (file.is_open() == false) {
    ERR(L"The file %ls can not be opened.", path_name.c_str());
    return false;
  }

  if (file.eof()) {
    ERR(L"File `%ls` is empty.", path_name.c_str());
    return false;
  }

  if (read_stream_block_n_parse<size_t>(
          tmp_ptr_array_buffers, tmp_ptr_last_buffer, tmp_block_size,
          this->p_file_buffer_size, this->p_file_buffer_shift_size, n_data,
          tmp_vector_buffers, file, L'\n') == false) {
    ERR(L"An error has been triggered from the "
        L"`read_stream_block_n_parse(ptr, ptr, %zu, %zu, %zu, ptr, vector, "
        L"wifstream, '\\n')` function, while reading the number of examples.",
        tmp_block_size, this->p_file_buffer_size,
        this->p_file_buffer_shift_size);
    return false;
  } else if (read_stream_block_n_parse<size_t>(
                 tmp_ptr_array_buffers, tmp_ptr_last_buffer, tmp_block_size,
                 this->p_file_buffer_size, this->p_file_buffer_shift_size,
                 tmp_number_outputs, tmp_vector_buffers, file,
                 L'\n') == false) {
    ERR(L"An error has been triggered from the "
        L"`read_stream_block_n_parse(ptr, ptr, %zu, %zu, %zu, ptr, vector, "
        L"wifstream, '\\n')` function, while reading the number of outputs.",
        tmp_block_size, this->p_file_buffer_size,
        this->p_file_buffer_shift_size);
    return false;
  } else if (read_stream_block_n_parse<size_t>(
                 tmp_ptr_array_buffers, tmp_ptr_last_buffer, tmp_block_size,
                 this->p_file_buffer_size, this->p_file_buffer_shift_size,
                 seq_w, tmp_vector_buffers, file, L'\n') == false) {
    ERR(L"An error has been triggered from the "
        L"`read_stream_block_n_parse(ptr, ptr, %zu, %zu, %zu, ptr, vector, "
        L"wifstream, '\\n')` function, while reading the number of recurrent "
        L"depth.",
        tmp_block_size, this->p_file_buffer_size,
        this->p_file_buffer_shift_size);
    return false;
  }

  tmp_ptr_array_outputs_array = new real const *[n_data];
  tmp_ptr_array_outputs = new real[tmp_number_outputs * n_data * seq_w];

  for (i = 0_UZ; i != n_data; ++i) {
    tmp_ptr_array_outputs_array[i] =
        tmp_ptr_array_outputs + i * tmp_number_outputs * seq_w;
  }

  for (i = 0_UZ; i != n_data; ++i) {
    for (t = 0_UZ; t != seq_w; ++t) {
      for (tmp_index = 0_UZ; tmp_index != tmp_number_outputs;
           ++tmp_index, ++tmp_ptr_array_outputs) {
        if (read_stream_block_n_parse<double>(
                tmp_ptr_array_buffers, tmp_ptr_last_buffer, tmp_block_size,
                this->p_file_buffer_size, this->p_file_buffer_shift_size,
                tmp_output, tmp_vector_buffers, file, L'\n') == false) {
          ERR(L"An error has been triggered from the "
              L"`read_stream_block_n_parse(ptr, ptr, %zu, %zu, %zu, ptr, "
              L"vector, wifstream, '\\n')` function, while reading data %zu "
              L"at %zu output.",
              tmp_block_size, this->p_file_buffer_size,
              this->p_file_buffer_shift_size, i, tmp_index);

          delete[](tmp_ptr_array_outputs_array[0]);
          delete[](tmp_ptr_array_outputs_array);

          return false;
        }

        *tmp_ptr_array_outputs = static_cast<real>(tmp_output);
      }
    }
  }

  if (file.fail()) {
    ERR(L"Logical error on i/o operation \"%ls\".", path_name.c_str());
    return false;
  }

  file.close();

  this->p_n_data_alloc = this->p_n_data = n_data;
  this->p_n_out = tmp_number_outputs;
  this->p_seq_w = seq_w;

  this->Ym = tmp_ptr_array_outputs_array;
  this->Y = tmp_ptr_array_outputs - n_data * tmp_number_outputs * seq_w;

  return true;
}

bool DatasetV1::Allocate__MNIST(std::wstring const &path_name) {
  int tmp_input_size, tmp_number_outputs, tmp_number_images, tmp_number_labels,
      tmp_number_rows, tmp_number_columns, tmp_index, i, tmp_magic_number;

  unsigned char tmp_input;

  real const **tmp_ptr_array_inputs_array(nullptr),
      **tmp_ptr_array_outputs_array(nullptr);
  real *tmp_ptr_array_inputs, *tmp_ptr_array_outputs;

  std::wstring const tmp_path_images(path_name + L".idx3-ubyte"),
      tmp_path_label(path_name + L".idx1-ubyte");

  if (path_exist(tmp_path_images) == false) {
    ERR(L"Could not find the following path `%ls`.", tmp_path_images.c_str());
    return false;
  } else if (path_exist(tmp_path_label) == false) {
    ERR(L"Could not find the following path `%ls`.", tmp_path_label.c_str());
    return false;
  }

  std::ifstream tmp_ifstream_images(CP_STR(tmp_path_images),
                                    std::ios::in | std::ios::binary),
      tmp_ifstream_labels(CP_STR(tmp_path_label),
                          std::ios::in | std::ios::binary);

  if (tmp_ifstream_images.is_open() == false) {
    ERR(L"The file %ls can not be opened.", tmp_path_images.c_str());
    return false;
  } else if (tmp_ifstream_images.eof()) {
    ERR(L"File `%ls` is empty.", tmp_path_images.c_str());
    return false;
  }

  if (tmp_ifstream_labels.is_open() == false) {
    ERR(L"The file %ls can not be opened.", tmp_path_label.c_str());
    return false;
  } else if (tmp_ifstream_labels.eof()) {
    ERR(L"File `%ls` is empty.", tmp_path_label.c_str());
    return false;
  }

  // MNIST image file.
  tmp_ifstream_images.read((char *)&tmp_magic_number, sizeof(int));
  tmp_magic_number = Math::reverse_int<int>(tmp_magic_number);

  if (tmp_magic_number != 2051) {
    ERR(L"Invalid MNIST image file! Magic number equal %d.", tmp_magic_number);

    return false;
  }

  tmp_ifstream_images.read((char *)&tmp_number_images, sizeof(int));
  tmp_number_images = Math::reverse_int<int>(tmp_number_images);

  tmp_ifstream_images.read((char *)&tmp_number_rows, sizeof(int));
  tmp_number_rows = Math::reverse_int<int>(tmp_number_rows);

  tmp_ifstream_images.read((char *)&tmp_number_columns, sizeof(int));
  tmp_number_columns = Math::reverse_int<int>(tmp_number_columns);

  tmp_input_size = tmp_number_rows * tmp_number_columns;

  tmp_ptr_array_inputs_array = new real const *[tmp_number_images];
  tmp_ptr_array_inputs = new real[tmp_number_images * tmp_input_size];

  for (i = 0; i != tmp_number_images; ++i) {
    tmp_ptr_array_inputs_array[i] = tmp_ptr_array_inputs;

    for (tmp_index = 0; tmp_index != tmp_input_size;
         ++tmp_index, ++tmp_ptr_array_inputs) {
      tmp_ifstream_images.read((char *)&tmp_input, sizeof(unsigned char));

      *tmp_ptr_array_inputs = static_cast<real>(tmp_input) / 255_r;
    }
  }

  tmp_ifstream_images.close();
  // |END| MNIST image file. |END|

  // MNIST label file.
  tmp_ifstream_labels.read((char *)&tmp_magic_number, sizeof(int));
  tmp_magic_number = Math::reverse_int<int>(tmp_magic_number);

  if (tmp_magic_number != 2049) {
    ERR(L"Invalid MNIST image file! Magic number equal %d.", tmp_magic_number);

    delete[](tmp_ptr_array_inputs_array[0]);
    delete[](tmp_ptr_array_inputs_array);

    return false;
  }

  tmp_ifstream_labels.read((char *)&tmp_number_labels, sizeof(int));
  tmp_number_labels = Math::reverse_int<int>(tmp_number_labels);

  if (tmp_number_images != tmp_number_labels) {
    ERR(L"The number of images (%d) differs with the number of labels (%d).",
        tmp_number_images, tmp_number_labels);

    delete[](tmp_ptr_array_inputs_array[0]);
    delete[](tmp_ptr_array_inputs_array);

    return false;
  }

  tmp_number_outputs = 10;

  tmp_ptr_array_outputs_array = new real const *[tmp_number_labels];
  tmp_ptr_array_outputs = new real[tmp_number_labels * tmp_number_outputs];
  memset(tmp_ptr_array_outputs, 0,
         static_cast<size_t>(tmp_number_labels * tmp_number_outputs) *
             sizeof(real));

  for (i = 0; i != tmp_number_labels; ++i) {
    tmp_ptr_array_outputs_array[i] = tmp_ptr_array_outputs;

    tmp_ifstream_labels.read((char *)&tmp_input, sizeof(unsigned char));

    tmp_ptr_array_outputs[tmp_input] = 1_r;

    tmp_ptr_array_outputs += tmp_number_outputs;
  }

  tmp_ifstream_labels.close();
  // |END| MNIST label file. |END|

  this->p_n_data_alloc = this->p_n_data =
      static_cast<size_t>(tmp_number_images);
  this->p_n_inp = static_cast<size_t>(tmp_input_size);
  this->p_n_out = static_cast<size_t>(tmp_number_outputs);
  this->p_seq_w = 1_UZ;

  this->Xm = tmp_ptr_array_inputs_array;
  this->X = tmp_ptr_array_inputs - tmp_number_images * tmp_input_size;
  this->Ym = tmp_ptr_array_outputs_array;
  this->Y = tmp_ptr_array_outputs - tmp_number_images * tmp_number_outputs;

  return true;
}

bool DatasetV1::Remove_Duplicate(void) {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");
    return false;
  }

  size_t tmp_new_number_examples(0_UZ), i, tmp_data_check_index, k(0_UZ), t;

  real *tmp_ptr_array_inputs(
      new real[this->p_n_data * this->p_n_inp * this->p_seq_w]),
      *tmp_ptr_array_outputs(
          new real[this->p_n_data * this->p_n_out * this->p_seq_w]);

  for (i = 0_UZ; i != this->p_n_data; ++i) {
    for (tmp_data_check_index = i + 1_UZ; tmp_data_check_index < this->p_n_data;
         ++tmp_data_check_index) {
      for (t = 0_UZ; t != this->p_seq_w; ++t) {
        // Check inputs duplications
        for (k = 0_UZ; k != this->p_n_inp; ++k)
          if (this->Xm[i][k] !=
              this->Xm[tmp_data_check_index][this->p_n_inp * t + k])
            break;

        // If duplicate.
        if (k == this->p_n_inp)
          break;
      }

      // If duplicate.
      if (k == this->p_n_inp)
        break;
    }

    // If not duplicate.
    if (k != this->p_n_inp) {
      // Store current inputs to a tempory array of inputs
      for (t = 0_UZ; t != this->p_seq_w; ++t)
        for (k = 0_UZ; k != this->p_n_inp; ++k)
          tmp_ptr_array_inputs[k] = this->Xm[i][this->p_n_inp * t + k];
      tmp_ptr_array_inputs += this->p_n_inp * this->p_seq_w;

      // Store current outputs to a tempory array of outputs
      for (t = 0_UZ; t != this->p_seq_w; ++t)
        for (k = 0_UZ; k != this->p_n_out; ++k)
          tmp_ptr_array_outputs[k] = this->Ym[i][this->p_n_out * t + k];
      tmp_ptr_array_outputs += this->p_n_out * this->p_seq_w;

      // Increment nData
      ++tmp_new_number_examples;
    }
  }

  // Reset pointer position to begining
  tmp_ptr_array_inputs -=
      tmp_new_number_examples * this->p_n_inp * this->p_seq_w;
  tmp_ptr_array_outputs -=
      tmp_new_number_examples * this->p_n_out * this->p_seq_w;

  if (this->p_n_data != tmp_new_number_examples) {
    SAFE_DELETE_ARRAY(this->X);
    SAFE_DELETE_ARRAY(this->Xm);
    SAFE_DELETE_ARRAY(this->Y);
    SAFE_DELETE_ARRAY(this->Ym);

    // Alloc
    this->Xm = new real const *[tmp_new_number_examples];
    this->X = tmp_ptr_array_inputs;

    this->Ym = new real const *[tmp_new_number_examples];
    this->Y = tmp_ptr_array_outputs;

    // Assign new data
    for (i = 0_UZ; i != tmp_new_number_examples; ++i) {
      this->Xm[i] = tmp_ptr_array_inputs;
      tmp_ptr_array_inputs += this->p_n_inp * this->p_seq_w;

      this->Ym[i] = tmp_ptr_array_outputs;
      tmp_ptr_array_outputs += this->p_n_out * this->p_seq_w;
    }

    this->p_n_data_alloc = this->p_n_data = tmp_new_number_examples;
  } else {
    delete[](tmp_ptr_array_inputs);
    delete[](tmp_ptr_array_outputs);
  }

  return true;
}

bool DatasetV1::Spliting_Dataset(size_t const desired_data_per_file_received,
                               std::wstring const &path_name) {
  size_t const tmp_number_files_to_create(static_cast<size_t>(
      ceil(static_cast<double>(this->p_n_data) /
           static_cast<double>(desired_data_per_file_received))));

  if (tmp_number_files_to_create == 1_UZ) {
    ERR(L"Can not generate only one file.");
    return false;
  } else if (path_name.find(L".dataset") == std::wstring::npos) {
    ERR(L"Can not find \".dataset\" in the path `%ls`.", path_name.c_str());
    return false;
  }

  std::wstring tmp_path, tmp_string_write;

  std::wofstream file;

  real const **tmp_ptr_array_inputs(this->Xm),
      **tmp_ptr_array_outputs(this->Ym);

  for (size_t tmp_data_per_file, i, t, tmp_index, tmp_file_index_shift(0_UZ),
       tmp_file_index(0_UZ);
       tmp_file_index != tmp_number_files_to_create; ++tmp_file_index) {
    tmp_data_per_file =
        tmp_file_index + 1_UZ != tmp_number_files_to_create
            ? desired_data_per_file_received
            : std::min<size_t>(desired_data_per_file_received,
                               this->p_n_data - desired_data_per_file_received *
                                                    tmp_file_index);

    tmp_path = path_name;
    tmp_path.erase(tmp_path.end() - 8, tmp_path.end());  // ".dataset"
    tmp_path += L"_" + std::to_wstring(tmp_file_index_shift++) + L".dataset";

    while (path_exist(tmp_path)) {
      tmp_path.erase(
          tmp_path.end() - (9 + std::to_wstring(tmp_file_index_shift).length()),
          tmp_path.end());
      tmp_path += L"_" + std::to_wstring(tmp_file_index_shift++) + L".dataset";
    }

    file.open(CP_STR(path_name),
              std::ios::out | std::ios::binary | std::ios::trunc);

    if (file.is_open() == false) {
      ERR(L"The file %ls can not be opened.", path_name.c_str());
      return false;
    }

    // Topology
    tmp_string_write = std::to_wstring(tmp_data_per_file) + L" " +
                       std::to_wstring(this->p_n_inp) + L" " +
                       std::to_wstring(this->p_n_out) + L" " +
                       std::to_wstring(this->p_seq_w);

    // Input & Output
    for (i = 0_UZ; i != tmp_data_per_file; ++i) {
      for (t = 0_UZ; t != this->p_seq_w; ++t) {
        // Input
        tmp_string_write += CRLF;
        for (tmp_index = 0_UZ; tmp_index != this->p_n_inp; ++tmp_index) {
          tmp_string_write += to_wstring(
              tmp_ptr_array_inputs[i][t * this->p_n_inp + tmp_index], 9u);

          if (tmp_index + 1_UZ != this->p_n_inp)
            tmp_string_write += L" ";
        }

        // Output
        tmp_string_write += CRLF;
        for (tmp_index = 0_UZ; tmp_index != this->p_n_out; ++tmp_index) {
          tmp_string_write += to_wstring(
              tmp_ptr_array_outputs[i][t * this->p_n_out + tmp_index], 9u);

          if (tmp_index + 1_UZ != this->p_n_out)
            tmp_string_write += L" ";
        }
      }
    }

    file.write(tmp_string_write.c_str(), static_cast<std::streamsize>(tmp_string_write.size()));

    file.flush();
    file.close();

    tmp_ptr_array_inputs += tmp_data_per_file;
    tmp_ptr_array_outputs += tmp_data_per_file;
  }

  return true;
}

bool DatasetV1::simulate_trading(Model *const model) {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");
    return false;
  } else if (this->valide_spec(model->n_inp, model->n_out, model->seq_w) ==
             false) {
    ERR(L"An error has been triggered from the "
        L"`valide_spec(%zu, %zu, %zu)` function.",
        model->n_inp, model->n_out, model->seq_w);
    return false;
  } else if (model->update_mem_batch_size(this->get_n_data()) == false) {
    ERR(L"An error has been triggered from the "
        L"`update_mem_batch_size(%zu)` function.",
        this->get_n_data());
    return false;
  }

  size_t const tmp_number_outputs(this->get_n_out()),
      tmp_timed_index(this->p_seq_w - 1_UZ), n_data(this->get_n_data()),
      tmp_maximum_batch_size(model->batch_size),
      tmp_number_batchs(static_cast<size_t>(
          ceil(static_cast<double>(n_data) /
               static_cast<double>(tmp_maximum_batch_size))));
  size_t tmp_batch_size, tmp_batch_index, i, k, tmp_number_same_sign(0_UZ);

  real tmp_desired_output, tmp_output;

  for (tmp_batch_index = 0_UZ; tmp_batch_index != tmp_number_batchs;
       ++tmp_batch_index) {
    tmp_batch_size = tmp_batch_index + 1_UZ != tmp_number_batchs
                         ? tmp_maximum_batch_size
                         : n_data - tmp_batch_index * tmp_maximum_batch_size;

    model->forward_pass(
        tmp_batch_size,
        this->Get__Input_Array() + tmp_batch_index * tmp_maximum_batch_size);

    for (i = 0_UZ; i != tmp_batch_size; ++i) {
      tmp_desired_output = 0_r;
      tmp_output = 0_r;

      for (k = 0_UZ; k != tmp_number_outputs; ++k) {
        tmp_desired_output +=
            this->get_out(tmp_batch_index * tmp_maximum_batch_size + i,
                          tmp_timed_index * tmp_number_outputs + k);

        tmp_output += cast(model->get_out(i, tmp_timed_index)[k]);
      }

      tmp_number_same_sign += static_cast<size_t>(
          Math::sign(tmp_output) == Math::sign(tmp_desired_output));
    }
  }

  INFO(L"");
  INFO(L"Report total trades: %zu", n_data);
  INFO(L"\tSucces: %f%%", static_cast<double>(tmp_number_same_sign) /
                              static_cast<double>(n_data) * 100.0);

  return true;
}

bool DatasetV1::replace_entries(DatasetV1 const *const dataset,
                              DATA::TYPE const data_type) {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");
    return false;
  } else if (dataset->p_n_data != this->p_n_data) {
    ERR(L"Source number data (%zu) differ from destination number data (%zu).",
        dataset->p_n_data, this->p_n_data);
    return false;
  } else if (dataset->get_seq_w() != this->p_seq_w) {
    ERR(L"Source recurrent depth (%zu) differ from destination recurrent depth "
        L"(%zu).",
        dataset->get_seq_w(), this->p_seq_w);
    return false;
  } else if (data_type > DATA::OUTPUT) {
    ERR(L"Type input (%d) is not managed in the function.", data_type);
    return false;
  } else if (this->_reference) {
    ERR(L"The dataset is allocate as refence.");
    return false;
  }

  size_t const tmp_input_size(data_type == DATA::INPUT ? dataset->get_n_inp()
                                                       : dataset->get_n_out());

  real const *const tmp_ptr_array_source_inputs(data_type == DATA::INPUT
                                                    ? dataset->get_inp(0_UZ)
                                                    : dataset->get_out(0_UZ));

  if (tmp_ptr_array_source_inputs == nullptr) {
    ERR(L"`tmp_ptr_array_source_inputs` is a nullptr.");
    return false;
  }

  real const **tmp_ptr_array_inputs_array;
  real *tmp_ptr_array_inputs;

  if (data_type == DATA::INPUT) {
    tmp_ptr_array_inputs = Mem::reallocate<real, false>(
        this->X, this->p_n_data * this->p_seq_w * tmp_input_size,
        this->p_n_data * this->p_seq_w * this->p_n_inp);

    this->X = tmp_ptr_array_inputs;

    tmp_ptr_array_inputs_array = this->Xm;

    this->p_n_inp = tmp_input_size;
  } else {
    tmp_ptr_array_inputs = Mem::reallocate<real, false>(
        this->Y, this->p_n_data * this->p_seq_w * tmp_input_size,
        this->p_n_data * this->p_seq_w * this->p_n_out);

    this->Y = tmp_ptr_array_inputs;

    tmp_ptr_array_inputs_array = this->Ym;

    this->p_n_out = tmp_input_size;
  }

  memcpy(tmp_ptr_array_inputs, tmp_ptr_array_source_inputs,
         this->p_n_data * this->p_seq_w * tmp_input_size * sizeof(real));

  for (size_t i(0_UZ); i != this->p_n_data; ++i)
    tmp_ptr_array_inputs_array[i] =
        tmp_ptr_array_inputs + i * tmp_input_size * this->p_seq_w;

  // TODO: Deep copy inputs/outputs.
  // ...
  // ...
  // ...

  return true;
}

bool DatasetV1::replace_entries(DatasetV1 const *const dataset,
                              Model *const model) {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");
    return false;
  } else if (this->p_n_data != dataset->p_n_data) {
    ERR(L"The number of data (%zu) differ from the number of data received as "
        L"argument (%zu).",
        this->p_n_data, dataset->p_n_data);
    return false;
  } else if (this->p_seq_w != dataset->p_seq_w) {
    ERR(L"The number of recurrent depth (%zu) differ from the number of "
        L"recurrent depth received as argument (%zu).",
        this->p_seq_w, dataset->p_seq_w);
    return false;
  } else if (dataset->valide_spec(model->n_inp, model->n_out, model->seq_w) ==
             false) {
    ERR(L"An error has been triggered from the "
        L"`valide_spec(%zu, %zu, %zu)` function.",
        model->n_inp, model->n_out, model->seq_w);
    return false;
  } else if (model->type != MODEL::AUTOENCODER) {
    ERR(L"The neural network (%ls) receive as argument need to be a %ls.",
        MODEL_NAME[model->type].c_str(),
        MODEL_NAME[MODEL::AUTOENCODER].c_str());
    return false;
  }

  auto tmp_Reset_IO_Mode(
      [tmp_use_first_layer_as_input = model->use_first_layer_as_input,
       tmp_use_last_layer_as_output = model->use_last_layer_as_output,
       &model]() -> bool {
        bool tmp_succes(true);

        if (model->Set__Input_Mode(tmp_use_first_layer_as_input) == false) {
          ERR(L"An error has been triggered from the "
              L"`Set__Input_Mode(%ls)` function.",
              to_wstring(tmp_use_first_layer_as_input).c_str());
          tmp_succes = false;
        }

        if (model->Set__Output_Mode(tmp_use_last_layer_as_output) == false) {
          ERR(L"An error has been triggered from the "
              L"`Set__Output_Mode(%ls)` function.",
              to_wstring(tmp_use_last_layer_as_output).c_str());
          tmp_succes = false;
        }

        return tmp_succes;
      });

  if (model->Set__Input_Mode(true) == false) {
    ERR(L"An error has been triggered from the "
        L"`Set__Input_Mode(true)` function.");
    return false;
  } else if (model->Set__Output_Mode(false) == false) {
    ERR(L"An error has been triggered from the "
        L"`Set__Output_Mode(false)` function.");

    if (tmp_Reset_IO_Mode() == false) {
      ERR(L"An error has been triggered from the "
          L"`tmp_Reset_IO_Mode()` function.");
      return false;
    }

    return false;
  } else if (this->p_n_inp != model->get_n_out()) {
    ERR(L"The number of input(s) (%zu) differ from the number of output(s) "
        L"from the autoencoder (%zu).",
        this->p_n_inp, model->get_n_out());

    if (tmp_Reset_IO_Mode() == false) {
      ERR(L"An error has been triggered from the "
          L"`tmp_Reset_IO_Mode()` function.");
      return false;
    }

    return false;
  }

  size_t const n_data(this->p_n_data),
      tmp_maximum_batch_size(model->batch_size),
      tmp_number_batchs(static_cast<size_t>(
          ceil(static_cast<double>(n_data) /
               static_cast<double>(tmp_maximum_batch_size))));
  size_t k, t, i, tmp_batch_size, tmp_batch_index;

  var const *Q;

  for (tmp_batch_index = 0_UZ; tmp_batch_index != tmp_number_batchs;
       ++tmp_batch_index) {
    tmp_batch_size = tmp_batch_index + 1_UZ != tmp_number_batchs
                         ? tmp_maximum_batch_size
                         : n_data - tmp_batch_index * tmp_maximum_batch_size;

    model->forward_pass(tmp_batch_size,
                        dataset->Xm + tmp_batch_index * tmp_maximum_batch_size);

    for (i = 0_UZ; i != tmp_batch_size; ++i)
      for (t = 0_UZ; t != this->p_seq_w; ++t) {
        Q = model->get_out(i, t);

        for (k = 0_UZ; k != this->p_n_inp; ++k)
          this->X[i * this->p_n_inp * this->p_seq_w + t * this->p_n_inp + k] =
              cast(Q[k]);
      }
  }

  if (tmp_Reset_IO_Mode() == false) {
    ERR(L"An error has been triggered from the "
        L"`tmp_Reset_IO_Mode()` function.");
    return false;
  }

  return true;
}

bool DatasetV1::Concat(DatasetV1 const *const dataset) {
  if (dataset->p_n_data == 0_UZ) {
    ERR(L"No data available from the source.");
    return false;
  } else if (this->valide_spec(dataset->get_n_inp(), dataset->get_n_out(),
                               dataset->get_seq_w()) == false) {
    ERR(L"An error has been triggered from the "
        L"`valide_spec(%zu, %zu, %zu)` function.",
        dataset->get_n_inp(), dataset->get_n_out(), dataset->get_seq_w());
    return false;
  } else if (dataset->get_seq_w() != this->p_seq_w) {
    ERR(L"Source recurrent depth (%zu) "
        L"differ from destination recurrent depth (%zu).",
        dataset->get_seq_w(), this->p_seq_w);
    return false;
  } else if (this->_reference) {
    ERR(L"The dataset is allocate as refence.");
    return false;
  }

  size_t const tmp_concat_number_examples(this->p_n_data + dataset->p_n_data);

  // Array inputs.
  real *tmp_ptr_array_inputs(Mem::reallocate(
      this->X, tmp_concat_number_examples * this->p_seq_w * this->p_n_inp,
      this->p_n_data * this->p_seq_w * this->p_n_inp));
  this->X = tmp_ptr_array_inputs;

  real const **tmp_ptr_array_inputs_array(
      Mem::reallocate_ptofpt<real const *, false>(
          this->Xm, tmp_concat_number_examples, this->p_n_data));
  this->Xm = tmp_ptr_array_inputs_array;

  memcpy(tmp_ptr_array_inputs + this->p_n_data * this->p_seq_w * this->p_n_inp,
         dataset->X,
         (tmp_concat_number_examples - this->p_n_data) * this->p_seq_w *
             this->p_n_inp * sizeof(real));
  // |END| Array inputs. |END|

  // Array outputs.
  real *tmp_ptr_array_outputs(Mem::reallocate(
      this->Y, tmp_concat_number_examples * this->p_seq_w * this->p_n_out,
      this->p_n_data * this->p_seq_w * this->p_n_out));
  this->Y = tmp_ptr_array_outputs;

  real const **tmp_ptr_array_outputs_array(
      Mem::reallocate_ptofpt<real const *, false>(
          this->Ym, tmp_concat_number_examples, this->p_n_data));
  this->Ym = tmp_ptr_array_outputs_array;

  memcpy(tmp_ptr_array_outputs + this->p_n_data * this->p_seq_w * this->p_n_out,
         dataset->Y,
         (tmp_concat_number_examples - this->p_n_data) * this->p_seq_w *
             this->p_n_out * sizeof(real));
  //|END| Array outputs. |END|

  this->p_n_data = tmp_concat_number_examples;

  for (size_t i(0_UZ); i != tmp_concat_number_examples; ++i) {
    tmp_ptr_array_inputs_array[i] =
        tmp_ptr_array_inputs + i * this->p_n_inp * this->p_seq_w;

    tmp_ptr_array_outputs_array[i] =
        tmp_ptr_array_outputs + i * this->p_n_out * this->p_seq_w;
  }

  return true;
}

bool DatasetV1::Save__Sequential_Input(
    size_t const number_recurrent_depth_received,
    std::wstring const &path_name) {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");
    return false;
  } else if (number_recurrent_depth_received <= 1_UZ) {
    ERR(L"Recurrent depth (%zu) need to be greater or equal 2.",
        number_recurrent_depth_received);
    return false;
  } else if (number_recurrent_depth_received > this->p_n_inp) {
    ERR(L"Recurrent depth (%zu) greater than the number of inputs (%zu).",
        number_recurrent_depth_received, this->p_n_inp);
    return false;
  }

  if (create_temp_file(path_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`create_temp_file(%ls)` function.",
        path_name.c_str());
    return false;
  }

  size_t const tmp_number_inputs_per_time_step(
      static_cast<size_t>(this->p_n_inp / number_recurrent_depth_received));
  size_t t, i, tmp_index;

  std::wstring tmp_string_write;

  std::wofstream file(CP_STR(path_name),
                      std::ios::out | std::ios::binary | std::ios::trunc);

  if (file.is_open() == false) {
    ERR(L"The file %ls can not be opened.", path_name.c_str());
    return false;
  }

  // Topology
  tmp_string_write = std::to_wstring(this->p_n_data) + L" " +
                     std::to_wstring(tmp_number_inputs_per_time_step) + L" " +
                     std::to_wstring(this->p_n_out) + L" " +
                     std::to_wstring(number_recurrent_depth_received);

  // Input & Output
  for (i = 0_UZ; i != this->p_n_data; ++i)
    for (t = 0_UZ; t != number_recurrent_depth_received; ++t) {
      // Input
      tmp_string_write += CRLF;
      for (tmp_index = 0_UZ; tmp_index != tmp_number_inputs_per_time_step;
           ++tmp_index) {
        tmp_string_write += to_wstring(
            this->Xm[i][t * tmp_number_inputs_per_time_step + tmp_index], 9u);

        if (tmp_index + 1_UZ != tmp_number_inputs_per_time_step)
          tmp_string_write += L" ";
      }

      // Output
      tmp_string_write += CRLF;
      for (tmp_index = 0_UZ; tmp_index != this->p_n_out; ++tmp_index) {
        tmp_string_write += to_wstring(
            this->Ym[i][(t % this->p_seq_w) * this->p_n_out + tmp_index], 9u);

        if (tmp_index + 1_UZ != this->p_n_out) tmp_string_write += L" ";
      }
    }

  file.write(tmp_string_write.c_str(), static_cast<std::streamsize>(tmp_string_write.size()));

  file.flush();
  file.close();

  if (delete_temp_file(path_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`delete_temp_file(%ls)` function.",
        path_name.c_str());
    return false;
  }

  return true;
}

bool DatasetV1::preprocess_minmax(size_t const data_start_index_received,
                                size_t const data_end_index_received,
                                real const minval,
                                real const maxval,
                                real const minimum_range_received,
                                real const maximum_range_received,
                                DATA::TYPE const data_type) {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");
    return false;
  } else if (data_start_index_received > data_end_index_received) {
    ERR(L"Start index (%zu) can not be greater than end index (%zu).",
        data_start_index_received, data_end_index_received);
    return false;
  } else if (minval > maxval) {
    ERR(L"Minimum value (%f) can not be greater than maximum value (%f).",
        minval, maxval);
    return false;
  } else if (minimum_range_received > maximum_range_received) {
    ERR(L"Minimum range (%f) can not be greater than maximum range (%f).",
        minimum_range_received, maximum_range_received);
    return false;
  } else if (this->_reference) {
    ERR(L"The dataset is allocate as refence.");
    return false;
  } else if (minval == minimum_range_received ||
             maxval == maximum_range_received) {
    return true;
  }

  if (data_type == DATA::INPUT) {
    if (this->_ptr_input_array_scaler__minimum_maximum == nullptr)
      this->_ptr_input_array_scaler__minimum_maximum =
          new ScalerMinMax[this->p_n_inp];
  } else if (data_type == DATA::OUTPUT) {
    if (this->_ptr_output_array_scaler__minimum_maximum == nullptr)
      this->_ptr_output_array_scaler__minimum_maximum =
          new ScalerMinMax[this->p_n_out];
  } else {
    ERR(L"Type input (%d) is not managed in the function.", data_type);
    return false;
  }

  size_t const tmp_input_size(data_type == DATA::INPUT ? this->p_n_inp
                                                       : this->p_n_out);
  size_t k;

  for (k = 0_UZ; k != tmp_input_size; ++k)
    if (this->preprocess_minmax(data_start_index_received,
                                data_end_index_received, k,
                                minval, maxval,
                                minimum_range_received, maximum_range_received,
                                data_type) == false) {
      ERR(L"An error has been triggered from the `preprocess_minmax"
          L"(%zu, %zu, %zu, %f, %f, %f, %f, %d)` function.",
          data_start_index_received, data_end_index_received, k,
          minval, maxval,
          minimum_range_received, maximum_range_received,
          data_type);
      return false;
    }

  return true;
}

bool DatasetV1::preprocess_minmax(size_t const data_start_index_received,
                                size_t const data_end_index_received,
                                size_t const input_index_received,
                                real const minval,
                                real const maxval,
                                real const minimum_range_received,
                                real const maximum_range_received,
                                DATA::TYPE const data_type) {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");
    return false;
  } else if (data_start_index_received > data_end_index_received) {
    ERR(L"Start index (%zu) can not be greater than end index (%zu).",
        data_start_index_received, data_end_index_received);
    return false;
  } else if (minval > maxval) {
    ERR(L"Minimum value (%f) can not be greater than maximum value (%f).",
        minval, maxval);
    return false;
  } else if (minimum_range_received > maximum_range_received) {
    ERR(L"Minimum range (%f) can not be greater than maximum range (%f).",
        minimum_range_received, maximum_range_received);
    return false;
  } else if (this->_reference) {
    ERR(L"The dataset is allocate as refence.");
    return false;
  } else if (minval == minimum_range_received ||
             maxval == maximum_range_received) {
    return true;
  }

  if (data_type == DATA::INPUT) {
    if (this->_ptr_input_array_scaler__minimum_maximum == nullptr)
      this->_ptr_input_array_scaler__minimum_maximum =
          new ScalerMinMax[this->p_n_inp];
  } else if (data_type == DATA::OUTPUT) {
    if (this->_ptr_output_array_scaler__minimum_maximum == nullptr)
      this->_ptr_output_array_scaler__minimum_maximum =
          new ScalerMinMax[this->p_n_out];
  } else {
    ERR(L"Type input (%d) is not managed in the function.", data_type);
    return false;
  }

  size_t const tmp_input_size(data_type == DATA::INPUT ? this->p_n_inp
                                                       : this->p_n_out);
  size_t i, t;

  if (input_index_received >= tmp_input_size) {
    ERR(L"Input index (%zu) overflow (%zu).", input_index_received,
        tmp_input_size);
    return false;
  }

  ScalerMinMax *const tmp_ptr_scaler__minimum_maximum(
      data_type == DATA::INPUT
          ? &this->_ptr_input_array_scaler__minimum_maximum
                 [input_index_received]
          : &this->_ptr_output_array_scaler__minimum_maximum
                 [input_index_received]);

  if (tmp_ptr_scaler__minimum_maximum == nullptr) {
    ERR(L"`tmp_ptr_scaler__minimum_maximum` is a nullptr.");
    return false;
  }

  real *const tmp_ptr_array_inputs(data_type == DATA::INPUT ? this->X
                                                            : this->Y);

  for (i = data_start_index_received; i != data_end_index_received; ++i)
    for (t = 0_UZ; t != this->p_seq_w; ++t)
      tmp_ptr_array_inputs[i * tmp_input_size * this->p_seq_w +
                           t * tmp_input_size + input_index_received] =
          (((tmp_ptr_array_inputs[i * tmp_input_size * this->p_seq_w +
                                  t * tmp_input_size + input_index_received] -
             minval) *
            (maximum_range_received - minimum_range_received)) /
           (maxval - minval)) +
          minimum_range_received;

  tmp_ptr_scaler__minimum_maximum->minval = minval;
  tmp_ptr_scaler__minimum_maximum->maxval = maxval;

  tmp_ptr_scaler__minimum_maximum->minrge = minimum_range_received;
  tmp_ptr_scaler__minimum_maximum->maxrge = maximum_range_received;

  tmp_ptr_scaler__minimum_maximum->str_index = data_start_index_received;
  tmp_ptr_scaler__minimum_maximum->end_index = data_end_index_received;

  return true;
}

bool DatasetV1::preprocess_minmax(real *const ptr_array_inputs_received,
                                DATA::TYPE const data_type) {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");
    return false;
  }

  if (data_type == DATA::INPUT) {
    if (this->_ptr_input_array_scaler__minimum_maximum == nullptr) {
      ERR(L"`_ptr_input_array_scaler__minimum_maximum` is a nullptr.");
      return false;
    }
  } else if (data_type == DATA::OUTPUT) {
    if (this->_ptr_output_array_scaler__minimum_maximum == nullptr) {
      ERR(L"`_ptr_output_array_scaler__minimum_maximum` is a nullptr.");
      return false;
    }
  } else {
    ERR(L"Type input (%d) is not managed in the function.", data_type);
    return false;
  }

  size_t const tmp_input_size(data_type == DATA::INPUT ? this->p_n_inp
                                                       : this->p_n_out);
  size_t k;

  for (k = 0_UZ; k != tmp_input_size; ++k)
    if (this->preprocess_minmax(k, ptr_array_inputs_received, data_type) ==
        false) {
      ERR(L"An error has been triggered from the "
          L"`preprocess_minmax(%zu, ptr, %d)` function.",
          k, data_type);
      return false;
    }

  return true;
}

bool DatasetV1::preprocess_minmax(size_t const input_index_received,
                                real *const ptr_array_inputs_received,
                                DATA::TYPE const data_type) {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");
    return false;
  }

  if (data_type == DATA::INPUT) {
    if (this->_ptr_input_array_scaler__minimum_maximum == nullptr) {
      ERR(L"`_ptr_input_array_scaler__minimum_maximum` is a nullptr.");
      return false;
    }
  } else if (data_type == DATA::OUTPUT) {
    if (this->_ptr_output_array_scaler__minimum_maximum == nullptr) {
      ERR(L"`_ptr_output_array_scaler__minimum_maximum` is a nullptr.");
      return false;
    }
  } else {
    ERR(L"Type input (%d) is not managed in the function.", data_type);
    return false;
  }

  size_t const tmp_input_size(data_type == DATA::INPUT ? this->p_n_inp
                                                       : this->p_n_out);
  size_t t;

  if (input_index_received >= tmp_input_size) {
    ERR(L"Input index (%zu) overflow (%zu).", input_index_received,
        tmp_input_size);
    return false;
  }

  ScalerMinMax *const tmp_ptr_scaler__minimum_maximum(
      data_type == DATA::INPUT
          ? &this->_ptr_input_array_scaler__minimum_maximum
                 [input_index_received]
          : &this->_ptr_output_array_scaler__minimum_maximum
                 [input_index_received]);

  if (tmp_ptr_scaler__minimum_maximum == nullptr) {
    ERR(L"`tmp_ptr_scaler__minimum_maximum` is a nullptr.");
    return false;
  }

  real const tmp_minimum_value(tmp_ptr_scaler__minimum_maximum->minval),
      tmp_maximum_value(tmp_ptr_scaler__minimum_maximum->maxval),
      tmp_minimum_range(tmp_ptr_scaler__minimum_maximum->minrge),
      tmp_maximum_range(tmp_ptr_scaler__minimum_maximum->maxrge);

  for (t = 0_UZ; t != this->p_seq_w; ++t)
    ptr_array_inputs_received[t * tmp_input_size + input_index_received] =
        (((ptr_array_inputs_received[t * tmp_input_size +
                                     input_index_received] -
           tmp_minimum_value) *
          (tmp_maximum_range - tmp_minimum_range)) /
         (tmp_maximum_value - tmp_minimum_value)) +
        tmp_minimum_range;

  return true;
}

bool DatasetV1::preprocess_minmax_inv(DATA::TYPE const data_type) {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");
    return false;
  } else if (this->_reference) {
    ERR(L"The dataset is allocate as refence.");
    return false;
  }

  if (data_type == DATA::INPUT) {
    if (this->_ptr_input_array_scaler__minimum_maximum == nullptr) {
      ERR(L"`_ptr_input_array_scaler__minimum_maximum` is a nullptr.");
      return false;
    }
  } else if (data_type == DATA::OUTPUT) {
    if (this->_ptr_output_array_scaler__minimum_maximum == nullptr) {
      ERR(L"`_ptr_output_array_scaler__minimum_maximum` is a nullptr.");
      return false;
    }
  } else {
    ERR(L"Type input (%d) is not managed in the function.", data_type);
    return false;
  }

  size_t const tmp_input_size(data_type == DATA::INPUT ? this->p_n_inp
                                                       : this->p_n_out);
  size_t k;

  for (k = 0_UZ; k != tmp_input_size; ++k)
    if (this->preprocess_minmax_inv(k, data_type) == false) {
      ERR(L"An error has been triggered from the "
          L"`preprocess_minmax_inv(%zu, %d)` function.",
          k, data_type);
      return false;
    }

  return true;
}

bool DatasetV1::preprocess_minmax_inv(size_t const input_index_received,
                                    DATA::TYPE const data_type) {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");
    return false;
  } else if (this->_reference) {
    ERR(L"The dataset is allocate as refence.");
    return false;
  }

  if (data_type == DATA::INPUT) {
    if (this->_ptr_input_array_scaler__minimum_maximum == nullptr) {
      ERR(L"`_ptr_input_array_scaler__minimum_maximum` is a nullptr.");
      return false;
    }
  } else if (data_type == DATA::OUTPUT) {
    if (this->_ptr_output_array_scaler__minimum_maximum == nullptr) {
      ERR(L"`_ptr_output_array_scaler__minimum_maximum` is a nullptr.");
      return false;
    }
  } else {
    ERR(L"Type input (%d) is not managed in the function.", data_type);
    return false;
  }

  size_t const tmp_input_size(data_type == DATA::INPUT ? this->p_n_inp
                                                       : this->p_n_out);
  size_t i, t;

  if (input_index_received >= tmp_input_size) {
    ERR(L"Input index (%zu) overflow (%zu).", input_index_received,
        tmp_input_size);
    return false;
  }

  ScalerMinMax *const tmp_ptr_scaler__minimum_maximum(
      data_type == DATA::INPUT
          ? &this->_ptr_input_array_scaler__minimum_maximum
                 [input_index_received]
          : &this->_ptr_output_array_scaler__minimum_maximum
                 [input_index_received]);

  if (tmp_ptr_scaler__minimum_maximum == nullptr) {
    ERR(L"`tmp_ptr_scaler__minimum_maximum` is a nullptr.");
    return false;
  } else if (tmp_ptr_scaler__minimum_maximum->str_index ==
             tmp_ptr_scaler__minimum_maximum->end_index) {
    return true;
  }

  real const tmp_minimum_value(tmp_ptr_scaler__minimum_maximum->minrge),
      tmp_maximum_value(tmp_ptr_scaler__minimum_maximum->maxrge),
      tmp_minimum_range(tmp_ptr_scaler__minimum_maximum->minval),
      tmp_maximum_range(tmp_ptr_scaler__minimum_maximum->maxval);

  if (tmp_minimum_value == tmp_minimum_range ||
      tmp_maximum_value == tmp_maximum_range)
    return true;

  size_t const tmp_data_end_index(tmp_ptr_scaler__minimum_maximum->end_index);

  real *const tmp_ptr_array_inputs(data_type == DATA::INPUT ? this->X
                                                            : this->Y);

  for (i = tmp_ptr_scaler__minimum_maximum->str_index; i != tmp_data_end_index;
       ++i)
    for (t = 0_UZ; t != this->p_seq_w; ++t)
      tmp_ptr_array_inputs[i * tmp_input_size * this->p_seq_w +
                           t * tmp_input_size + input_index_received] =
          (((tmp_ptr_array_inputs[i * tmp_input_size * this->p_seq_w +
                                  t * tmp_input_size + input_index_received] -
             tmp_minimum_value) *
            (tmp_maximum_range - tmp_minimum_range)) /
           (tmp_maximum_value - tmp_minimum_value)) +
          tmp_minimum_range;

  tmp_ptr_scaler__minimum_maximum->str_index = 0_UZ;
  tmp_ptr_scaler__minimum_maximum->end_index = 0_UZ;

  tmp_ptr_scaler__minimum_maximum->minval = 0_r;
  tmp_ptr_scaler__minimum_maximum->maxval = 1_r;

  tmp_ptr_scaler__minimum_maximum->minrge = 0_r;
  tmp_ptr_scaler__minimum_maximum->maxrge = 1_r;

  return true;
}

bool DatasetV1::preprocess_minmax_inv(real *const ptr_array_inputs_received,
                                    DATA::TYPE const data_type) {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");
    return false;
  }

  if (data_type == DATA::INPUT) {
    if (this->_ptr_input_array_scaler__minimum_maximum == nullptr) {
      ERR(L"`_ptr_input_array_scaler__minimum_maximum` is a nullptr.");
      return false;
    }
  } else if (data_type == DATA::OUTPUT) {
    if (this->_ptr_output_array_scaler__minimum_maximum == nullptr) {
      ERR(L"`_ptr_output_array_scaler__minimum_maximum` is a nullptr.");
      return false;
    }
  } else {
    ERR(L"Type input (%d) is not managed in the function.", data_type);
    return false;
  }

  size_t const tmp_input_size(data_type == DATA::INPUT ? this->p_n_inp
                                                       : this->p_n_out);
  size_t k;

  for (k = 0_UZ; k != tmp_input_size; ++k)
    if (this->preprocess_minmax_inv(k, ptr_array_inputs_received, data_type) ==
        false) {
      ERR(L"An error has been triggered from the "
          L"`preprocess_minmax_inv(%zu, ptr, %d)` function.",
          k, data_type);
      return false;
    }

  return true;
}

bool DatasetV1::preprocess_minmax_inv(size_t const input_index_received,
                                    real *const ptr_array_inputs_received,
                                    DATA::TYPE const data_type) {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");
    return false;
  }

  if (data_type == DATA::INPUT) {
    if (this->_ptr_input_array_scaler__minimum_maximum == nullptr) {
      ERR(L"`_ptr_input_array_scaler__minimum_maximum` is a nullptr.");
      return false;
    }
  } else if (data_type == DATA::OUTPUT) {
    if (this->_ptr_output_array_scaler__minimum_maximum == nullptr) {
      ERR(L"`_ptr_output_array_scaler__minimum_maximum` is a nullptr.");
      return false;
    }
  } else {
    ERR(L"Type input (%d) is not managed in the function.", data_type);
    return false;
  }

  size_t const tmp_input_size(data_type == DATA::INPUT ? this->p_n_inp
                                                       : this->p_n_out);
  size_t t;

  if (input_index_received >= tmp_input_size) {
    ERR(L"Input index (%zu) overflow (%zu).", input_index_received,
        tmp_input_size);
    return false;
  }

  ScalerMinMax *const tmp_ptr_scaler__minimum_maximum(
      data_type == DATA::INPUT
          ? &this->_ptr_input_array_scaler__minimum_maximum
                 [input_index_received]
          : &this->_ptr_output_array_scaler__minimum_maximum
                 [input_index_received]);

  if (tmp_ptr_scaler__minimum_maximum == nullptr) {
    ERR(L"`tmp_ptr_scaler__minimum_maximum` is a nullptr.");
    return false;
  } else if (tmp_ptr_scaler__minimum_maximum->str_index ==
             tmp_ptr_scaler__minimum_maximum->end_index) {
    return true;
  }

  real const tmp_minimum_value(tmp_ptr_scaler__minimum_maximum->minrge),
      tmp_maximum_value(tmp_ptr_scaler__minimum_maximum->maxrge),
      tmp_minimum_range(tmp_ptr_scaler__minimum_maximum->minval),
      tmp_maximum_range(tmp_ptr_scaler__minimum_maximum->maxval);

  if (tmp_minimum_value == tmp_minimum_range ||
      tmp_maximum_value == tmp_maximum_range)
    return true;

  for (t = 0_UZ; t != this->p_seq_w; ++t)
    ptr_array_inputs_received[t * tmp_input_size + input_index_received] =
        (((ptr_array_inputs_received[t * tmp_input_size +
                                     input_index_received] -
           tmp_minimum_value) *
          (tmp_maximum_range - tmp_minimum_range)) /
         (tmp_maximum_value - tmp_minimum_value)) +
        tmp_minimum_range;

  return true;
}

bool DatasetV1::Preprocessing__Zero_Centered(
    size_t const data_start_index_received,
    size_t const data_end_index_received, real const multiplier_received,
    DATA::TYPE const data_type) {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");

    return false;
  } else if (data_start_index_received > data_end_index_received) {
    ERR(L"Start index (%zu) can not be greater than end index (%zu).",
        data_start_index_received, data_end_index_received);

    return false;
  } else if (this->_reference) {
    ERR(L"The dataset is allocate as refence.");

    return false;
  }

  if (data_type == DATA::INPUT) {
    if (this->_ptr_input_array_scaler__zero_centered == nullptr)
      this->_ptr_input_array_scaler__zero_centered =
          new ScalerZeroCentered[this->p_n_inp];
  } else if (data_type == DATA::OUTPUT) {
    if (this->_ptr_output_array_scaler__zero_centered == nullptr)
      this->_ptr_output_array_scaler__zero_centered =
          new ScalerZeroCentered[this->p_n_out];
  } else {
    ERR(L"Type input (%d) is not managed in the function.", data_type);
    return false;
  }

  size_t const tmp_input_size(data_type == DATA::INPUT ? this->p_n_inp
                                                       : this->p_n_out);
  size_t k;

  for (k = 0_UZ; k != tmp_input_size; ++k)
    if (this->Preprocessing__Zero_Centered(
            data_start_index_received, data_end_index_received, k,
            multiplier_received, data_type) == false) {
      ERR(L"An error has been triggered from the "
          L"`Preprocessing__Zero_Centered(%zu, %zu, %zu, %f, %d)` function.",
          data_start_index_received, data_end_index_received, k,
          multiplier_received, data_type);
      return false;
    }

  return true;
}

bool DatasetV1::Preprocessing__Zero_Centered(
    size_t const data_start_index_received,
    size_t const data_end_index_received, size_t const input_index_received,
    real const multiplier_received, DATA::TYPE const data_type) {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");
    return false;
  } else if (data_start_index_received > data_end_index_received) {
    ERR(L"Start index (%zu) can not be greater than end index (%zu).",
        data_start_index_received, data_end_index_received);
    return false;
  } else if (this->_reference) {
    ERR(L"The dataset is allocate as refence.");
    return false;
  }

  if (data_type == DATA::INPUT) {
    if (this->_ptr_input_array_scaler__zero_centered == nullptr)
      this->_ptr_input_array_scaler__zero_centered =
          new ScalerZeroCentered[this->p_n_inp];
  } else if (data_type == DATA::OUTPUT) {
    if (this->_ptr_output_array_scaler__zero_centered == nullptr)
      this->_ptr_output_array_scaler__zero_centered =
          new ScalerZeroCentered[this->p_n_out];
  } else {
    ERR(L"Type input (%d) is not managed in the function.", data_type);
    return false;
  }

  size_t const tmp_input_size(data_type == DATA::INPUT ? this->p_n_inp
                                                       : this->p_n_out);
  size_t i, t;

  if (input_index_received >= tmp_input_size) {
    ERR(L"Input index (%zu) overflow (%zu).", input_index_received,
        tmp_input_size);
    return false;
  }

  ScalerZeroCentered *const tmp_ptr_scaler__zero_centered(
      data_type == DATA::INPUT
          ? &this->_ptr_input_array_scaler__zero_centered[input_index_received]
          : &this->_ptr_output_array_scaler__zero_centered
                 [input_index_received]);

  if (tmp_ptr_scaler__zero_centered == nullptr) {
    ERR(L"`tmp_ptr_scaler__zero_centered` is a nullptr.");
    return false;
  }

  real *const tmp_ptr_array_inputs(data_type == DATA::INPUT ? this->X
                                                            : this->Y);

  for (i = data_start_index_received; i != data_end_index_received; ++i)
    for (t = 0_UZ; t != this->p_seq_w; ++t)
      tmp_ptr_array_inputs[i * tmp_input_size * this->p_seq_w +
                           t * tmp_input_size + input_index_received] *=
          multiplier_received;

  tmp_ptr_scaler__zero_centered->str_index = data_start_index_received;
  tmp_ptr_scaler__zero_centered->end_index = data_end_index_received;

  tmp_ptr_scaler__zero_centered->multiplier = multiplier_received;

  return true;
}

bool DatasetV1::Preprocessing__Zero_Centered(
    real *const ptr_array_inputs_received, DATA::TYPE const data_type) {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");
    return false;
  }

  if (data_type == DATA::INPUT) {
    if (this->_ptr_input_array_scaler__zero_centered == nullptr) {
      ERR(L"`_ptr_input_array_scaler__zero_centered` is a nullptr.");
      return false;
    }
  } else if (data_type == DATA::OUTPUT) {
    if (this->_ptr_output_array_scaler__zero_centered == nullptr) {
      ERR(L"`_ptr_output_array_scaler__zero_centered` is a nullptr.");
      return false;
    }
  } else {
    ERR(L"Type input (%d) is not managed in the function.", data_type);
    return false;
  }

  size_t const tmp_input_size(data_type == DATA::INPUT ? this->p_n_inp
                                                       : this->p_n_out);
  size_t k;

  for (k = 0_UZ; k != tmp_input_size; ++k)
    if (this->Preprocessing__Zero_Centered(k, ptr_array_inputs_received,
                                           data_type) == false) {
      ERR(L"An error has been triggered from the "
          L"`Preprocessing__Zero_Centered(%zu, ptr, %d)` function.",
          k, data_type);
      return false;
    }

  return true;
}

bool DatasetV1::Preprocessing__Zero_Centered(
    size_t const input_index_received, real *const ptr_array_inputs_received,
    DATA::TYPE const data_type) {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");
    return false;
  }

  if (data_type == DATA::INPUT) {
    if (this->_ptr_input_array_scaler__zero_centered == nullptr) {
      ERR(L"`_ptr_input_array_scaler__zero_centered` is a nullptr.");
      return false;
    }
  } else if (data_type == DATA::OUTPUT) {
    if (this->_ptr_output_array_scaler__zero_centered == nullptr) {
      ERR(L"`_ptr_output_array_scaler__zero_centered` is a nullptr.");
      return false;
    }
  } else {
    ERR(L"Type input (%d) is not managed in the function.", data_type);
    return false;
  }

  size_t const tmp_input_size(data_type == DATA::INPUT ? this->p_n_inp
                                                       : this->p_n_out);
  size_t t;

  if (input_index_received >= tmp_input_size) {
    ERR(L"Input index (%zu) overflow (%zu).", input_index_received,
        tmp_input_size);
    return false;
  }

  ScalerZeroCentered *const tmp_ptr_scaler__zero_centered(
      data_type == DATA::INPUT
          ? &this->_ptr_input_array_scaler__zero_centered[input_index_received]
          : &this->_ptr_output_array_scaler__zero_centered
                 [input_index_received]);

  if (tmp_ptr_scaler__zero_centered == nullptr) {
    ERR(L"`tmp_ptr_scaler__zero_centered` is a nullptr.");
    return false;
  } else if (tmp_ptr_scaler__zero_centered->str_index ==
             tmp_ptr_scaler__zero_centered->end_index) {
    return true;
  }

  real const tmp_multiplier(tmp_ptr_scaler__zero_centered->multiplier);

  if (tmp_multiplier == 1_r)
    return true;

  for (t = 0_UZ; t != this->p_seq_w; ++t)
    ptr_array_inputs_received[t * tmp_input_size + input_index_received] *=
        tmp_multiplier;

  return true;
}

bool DatasetV1::Preprocessing__Zero_Centered_Inverse(DATA::TYPE const data_type) {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");
    return false;
  } else if (this->_reference) {
    ERR(L"The dataset is allocate as refence.");
    return false;
  }

  if (data_type == DATA::INPUT) {
    if (this->_ptr_input_array_scaler__zero_centered == nullptr) {
      ERR(L"`_ptr_input_array_scaler__zero_centered` is a nullptr.");
      return false;
    }
  } else if (data_type == DATA::OUTPUT) {
    if (this->_ptr_output_array_scaler__zero_centered == nullptr) {
      ERR(L"`_ptr_output_array_scaler__zero_centered` is a nullptr.");
      return false;
    }
  } else {
    ERR(L"Type input (%d) is not managed in the function.", data_type);
    return false;
  }

  size_t const tmp_input_size(data_type == DATA::INPUT ? this->p_n_inp
                                                       : this->p_n_out);
  size_t k;

  for (k = 0_UZ; k != tmp_input_size; ++k)
    if (this->Preprocessing__Zero_Centered_Inverse(k, data_type) == false) {
      ERR(L"An error has been triggered from the "
          L"`preprocess_minmax_inv(%zu, %d)` function.",
          k, data_type);
      return false;
    }

  return true;
}

bool DatasetV1::Preprocessing__Zero_Centered_Inverse(
    size_t const input_index_received, DATA::TYPE const data_type) {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");
    return false;
  } else if (this->_reference) {
    ERR(L"The dataset is allocate as refence.");
    return false;
  }

  if (data_type == DATA::INPUT) {
    if (this->_ptr_input_array_scaler__zero_centered == nullptr) {
      ERR(L"`_ptr_input_array_scaler__zero_centered` is a nullptr.");
      return false;
    }
  } else if (data_type == DATA::OUTPUT) {
    if (this->_ptr_output_array_scaler__zero_centered == nullptr) {
      ERR(L"`_ptr_output_array_scaler__zero_centered` is a nullptr.");
      return false;
    }
  } else {
    ERR(L"Type input (%d) is not managed in the function.", data_type);
    return false;
  }

  size_t const tmp_input_size(data_type == DATA::INPUT ? this->p_n_inp
                                                       : this->p_n_out);
  size_t i, t;

  if (input_index_received >= tmp_input_size) {
    ERR(L"Input index (%zu) overflow (%zu).", input_index_received,
        tmp_input_size);
    return false;
  }

  ScalerZeroCentered *const tmp_ptr_scaler__zero_centered(
      data_type == DATA::INPUT
          ? &this->_ptr_input_array_scaler__zero_centered[input_index_received]
          : &this->_ptr_output_array_scaler__zero_centered
                 [input_index_received]);

  if (tmp_ptr_scaler__zero_centered == nullptr) {
    ERR(L"`tmp_ptr_scaler__zero_centered` is a nullptr.");
    return false;
  } else if (tmp_ptr_scaler__zero_centered->str_index ==
             tmp_ptr_scaler__zero_centered->end_index) {
    return true;
  }

  real const tmp_multiplier(1_r /
                            tmp_ptr_scaler__zero_centered->multiplier);

  if (tmp_multiplier == 1_r)
    return true;

  size_t const tmp_data_end_index(tmp_ptr_scaler__zero_centered->end_index);

  real *const tmp_ptr_array_inputs(data_type == DATA::INPUT ? this->X
                                                            : this->Y);

  for (i = tmp_ptr_scaler__zero_centered->str_index; i != tmp_data_end_index;
       ++i)
    for (t = 0_UZ; t != this->p_seq_w; ++t)
      tmp_ptr_array_inputs[i * tmp_input_size * this->p_seq_w +
                           t * tmp_input_size + input_index_received] *=
          tmp_multiplier;

  tmp_ptr_scaler__zero_centered->str_index = 0_UZ;
  tmp_ptr_scaler__zero_centered->end_index = 0_UZ;

  tmp_ptr_scaler__zero_centered->multiplier = 1_r;

  return true;
}

bool DatasetV1::preprocess_modwt(size_t const desired_J_level_received,
                               DATA::TYPE const data_type) {
  if (this->p_n_data <= 1_UZ) {
    ERR(L"No enought data available.");
    return false;
  } else if (this->p_seq_w == 0_UZ) {
    ERR(L"Recurrent depth can not be equal to zero.");
    return false;
  } else if (this->_reference) {
    ERR(L"The dataset is allocate as refence.");
    return false;
  } else if (desired_J_level_received == 0_UZ) {
    return true;
  }

  if (data_type == DATA::INPUT) {
    if (this->Xm_coeff == nullptr) {
      this->Xm_coeff = new real *[this->p_n_inp * this->p_seq_w];
      Mem::fill_null<real *>(this->Xm_coeff,
                                this->Xm_coeff + this->p_n_inp * this->p_seq_w);
    }

    if (this->Xm_coeff_size == nullptr) {
      this->Xm_coeff_size = new size_t[this->p_n_inp];
      memset(this->Xm_coeff_size, 0, this->p_n_inp * sizeof(size_t));
    }
  } else if (data_type == DATA::OUTPUT) {
    if (this->Ym_coeff == nullptr) {
      this->Ym_coeff = new real *[this->p_n_out * this->p_seq_w];
      Mem::fill_null<real *>(this->Ym_coeff,
                                this->Ym_coeff + this->p_n_out * this->p_seq_w);
    }

    if (this->Ym_coeff_size == nullptr) {
      this->Ym_coeff_size = new size_t[this->p_n_out];
      memset(this->Ym_coeff_size, 0, this->p_n_out * sizeof(size_t));
    }
  } else {
    ERR(L"Type input (%d) is not managed in the function.", data_type);
    return false;
  }

  size_t const tmp_input_size(data_type == DATA::INPUT ? this->p_n_inp
                                                       : this->p_n_out);
  size_t k;

  for (k = 0_UZ; k != tmp_input_size; ++k)
    if (this->preprocess_modwt(k, desired_J_level_received, data_type) ==
        false) {
      ERR(L"An error has been triggered from the "
          L"`preprocess_modwt(%zu, %zu, %d)` function.",
          k, desired_J_level_received, data_type);
      return false;
    }

  return true;
}

bool DatasetV1::preprocess_modwt(size_t const input_index_received,
                               size_t const desired_J_level_received,
                               DATA::TYPE const data_type) {
  if (this->p_n_data <= 1_UZ) {
    ERR(L"No enought data available.");
    return false;
  } else if (this->p_seq_w == 0_UZ) {
    ERR(L"Recurrent depth can not be equal to zero.");
    return false;
  } else if (this->_reference) {
    ERR(L"The dataset is allocate as refence.");
    return false;
  } else if (desired_J_level_received == 0_UZ) {
    return true;
  }

  if (data_type == DATA::INPUT) {
    if (this->Xm_coeff == nullptr) {
      this->Xm_coeff = new real *[this->p_n_inp * this->p_seq_w];
      Mem::fill_null<real *>(this->Xm_coeff,
                                this->Xm_coeff + this->p_n_inp * this->p_seq_w);
    }

    if (this->Xm_coeff_size == nullptr) {
      this->Xm_coeff_size = new size_t[this->p_n_inp];
      memset(this->Xm_coeff_size, 0, this->p_n_inp * sizeof(size_t));
    }
  } else if (data_type == DATA::OUTPUT) {
    if (this->Ym_coeff == nullptr) {
      this->Ym_coeff = new real *[this->p_n_out * this->p_seq_w];
      Mem::fill_null<real *>(this->Ym_coeff,
                                this->Ym_coeff + this->p_n_out * this->p_seq_w);
    }

    if (this->Ym_coeff_size == nullptr) {
      this->Ym_coeff_size = new size_t[this->p_n_out];
      memset(this->Ym_coeff_size, 0, this->p_n_out * sizeof(size_t));
    }
  } else {
    ERR(L"Type input (%d) is not managed in the function.", data_type);
    return false;
  }

  size_t const tmp_input_size(data_type == DATA::INPUT ? this->p_n_inp
                                                       : this->p_n_out);

  if (input_index_received >= tmp_input_size) {
    ERR(L"Input index (%zu) overflow (%zu).", input_index_received,
        tmp_input_size);
    return false;
  }

  size_t const tmp_J_level(
      std::min(this->MODWT__J_Level_Maximum(), desired_J_level_received));
  size_t tmp_coefficient_matrix_size, i, t;

  real *const tmp_ptr_array_inputs(data_type == DATA::INPUT ? this->X
                                                            : this->Y),
      *tmp_ptr_array_inputs_preproced, *tmp_ptr_array_smooth_coefficients;

  if (tmp_ptr_array_inputs == nullptr) {
    ERR(L"`tmp_ptr_array_inputs` is a nullptr.");
    return false;
  }

  tmp_ptr_array_inputs_preproced = new real[this->p_n_data];

  for (t = 0_UZ; t != this->p_seq_w; ++t) {
    real *&tmp_ptr_coefficient_matrix(
        data_type == DATA::INPUT
            ? this->Xm_coeff[input_index_received * this->p_seq_w + t]
            : this->Ym_coeff[input_index_received * this->p_seq_w + t]);

    tmp_coefficient_matrix_size =
        data_type == DATA::INPUT ? this->Xm_coeff_size[input_index_received]
                                 : this->Ym_coeff_size[input_index_received];

    // Get timed input.
    for (i = 0_UZ; i != this->p_n_data; ++i)
      tmp_ptr_array_inputs_preproced[i] =
          tmp_ptr_array_inputs[i * tmp_input_size * this->p_seq_w +
                               t * tmp_input_size + input_index_received];

    if (Math::modwt(this->p_n_data, tmp_coefficient_matrix_size,
                    tmp_ptr_array_inputs_preproced, tmp_ptr_coefficient_matrix,
                    tmp_J_level) == false) {
      ERR(L"An error has been triggered from the "
          L"`modwt(%zu, %zu, ptr, ptr, %zu)` function.",
          this->p_n_data, tmp_coefficient_matrix_size, tmp_J_level);
      return false;
    }

    // Set timed input.
    tmp_ptr_array_smooth_coefficients =
        tmp_ptr_coefficient_matrix + tmp_J_level * this->p_n_data;

    for (i = 0_UZ; i != this->p_n_data; ++i)
      tmp_ptr_array_inputs[i * tmp_input_size * this->p_seq_w +
                           t * tmp_input_size + input_index_received] =
          tmp_ptr_array_smooth_coefficients[i];
  }

  delete[](tmp_ptr_array_inputs_preproced);

  if (data_type == DATA::INPUT)
    this->Xm_coeff_size[input_index_received] = tmp_coefficient_matrix_size;
  else
    this->Ym_coeff_size[input_index_received] = tmp_coefficient_matrix_size;

  return true;
}

bool DatasetV1::preprocess_modwt(size_t const input_index_received,
                               real *const ptr_array_inputs_received,
                               DATA::TYPE const data_type) {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");
    return false;
  } else if (this->p_seq_w == 0_UZ) {
    ERR(L"Recurrent depth can not be equal to zero.");
    return false;
  }

  if (data_type == DATA::INPUT) {
    if (this->Xm_coeff == nullptr) {
      ERR(L"`Xm_coeff` is a nullptr.");
      return false;
    }

    if (this->Xm_coeff_size == nullptr) {
      ERR(L"`Xm_coeff_size` is a nullptr.");
      return false;
    }
  } else if (data_type == DATA::OUTPUT) {
    if (this->Ym_coeff == nullptr) {
      ERR(L"`Ym_coeff` is a nullptr.");
      return false;
    }

    if (this->Ym_coeff_size == nullptr) {
      ERR(L"`Ym_coeff_size` is a nullptr.");
      return false;
    }
  } else {
    ERR(L"Type input (%d) is not managed in the function.", data_type);
    return false;
  }

  size_t const tmp_input_size(data_type == DATA::INPUT ? this->p_n_inp
                                                       : this->p_n_out);

  if (input_index_received >= tmp_input_size) {
    ERR(L"Input index (%zu) overflow (%zu).", input_index_received,
        tmp_input_size);
    return false;
  }

  size_t const tmp_batch_size(this->p_n_data + 1_UZ),
      tmp_input_coefficient_matrix_size(
          data_type == DATA::INPUT ? this->Xm_coeff_size[input_index_received]
                                   : this->Ym_coeff_size[input_index_received]);
  size_t tmp_J_level(tmp_input_coefficient_matrix_size / this->p_n_data),
      tmp_coefficient_matrix_size, i, t, tmp_j_index;

  real const *tmp_ptr_source_coefficient_matrix;
  real *tmp_ptr_coefficient_matrix, *tmp_ptr_array_inputs_preproced,
      *tmp_ptr_array_smooth_coefficients;

  // Valid input index.
  if (tmp_J_level == 0_UZ)
    return true;
  else
    --tmp_J_level;
  // |END| Valid input index. |END|

  tmp_ptr_array_inputs_preproced = new real[tmp_batch_size];

  for (t = 0_UZ; t != this->p_seq_w; ++t) {
    tmp_ptr_source_coefficient_matrix =
        (data_type == DATA::INPUT
             ? this->Xm_coeff[input_index_received * this->p_seq_w + t]
             : this->Ym_coeff[input_index_received * this->p_seq_w + t]);
    tmp_ptr_coefficient_matrix = nullptr;

    tmp_coefficient_matrix_size = 0_UZ;

    // Get timed input from dataset.
    for (i = 0_UZ; i != this->p_n_data; ++i)
      tmp_ptr_array_inputs_preproced[i] = tmp_ptr_source_coefficient_matrix[i];
    for (tmp_j_index = 1_UZ; tmp_j_index != tmp_J_level + 1_UZ; ++tmp_j_index)
      for (i = 0_UZ; i != this->p_n_data; ++i)
        tmp_ptr_array_inputs_preproced[i] +=
            tmp_ptr_source_coefficient_matrix[tmp_j_index * this->p_n_data + i];
    // |END| Get timed input from dataset. |END|

    // Get timed input from arguments.
    tmp_ptr_array_inputs_preproced[i] =
        ptr_array_inputs_received[t * tmp_input_size + input_index_received];

    if (Math::modwt(tmp_batch_size, tmp_coefficient_matrix_size,
                    tmp_ptr_array_inputs_preproced, tmp_ptr_coefficient_matrix,
                    tmp_J_level) == false) {
      ERR(L"An error has been triggered from the "
          L"`modwt(%zu, %zu, ptr, ptr, %zu)` function.",
          tmp_batch_size, tmp_coefficient_matrix_size, tmp_J_level);
      return false;
    }

    // shift array for continious access.
    tmp_ptr_array_smooth_coefficients =
        tmp_ptr_coefficient_matrix + tmp_J_level * tmp_batch_size;

    // Set timed input from arguments.
    ptr_array_inputs_received[t * tmp_input_size + input_index_received] =
        tmp_ptr_array_smooth_coefficients[i];

    delete[](tmp_ptr_coefficient_matrix);
  }

  // Delete tempory inputs storage.
  delete[](tmp_ptr_array_inputs_preproced);
  // |END| Delete tempory inputs storage. |END|

  return true;
}

bool DatasetV1::preprocess_modwt_inv(DATA::TYPE const data_type) {
  if (this->p_n_data <= 1_UZ) {
    ERR(L"No enought data available.");
    return false;
  } else if (this->p_seq_w == 0_UZ) {
    ERR(L"Recurrent depth can not be equal to zero.");
    return false;
  } else if (this->_reference) {
    ERR(L"The dataset is allocate as refence.");
    return false;
  }

  if (data_type == DATA::INPUT) {
    if (this->Xm_coeff == nullptr) {
      ERR(L"`Xm_coeff` is a nullptr.");
      return false;
    }

    if (this->Xm_coeff_size == nullptr) {
      ERR(L"`Xm_coeff_size` is a nullptr.");
      return false;
    }
  } else if (data_type == DATA::OUTPUT) {
    if (this->Ym_coeff == nullptr) {
      ERR(L"`Ym_coeff` is a nullptr.");
      return false;
    }

    if (this->Ym_coeff_size == nullptr) {
      ERR(L"`Ym_coeff_size` is a nullptr.");
      return false;
    }
  } else {
    ERR(L"Type input (%d) is not managed in the function.", data_type);
    return false;
  }

  size_t const tmp_input_size(data_type == DATA::INPUT ? this->p_n_inp
                                                       : this->p_n_out);
  size_t k;

  for (k = 0_UZ; k != tmp_input_size; ++k)
    if (this->preprocess_modwt_inv(k, data_type) == false) {
      ERR(L"An error has been triggered from the "
          L"`preprocess_modwt_inv(%zu, %d)` function.",
          k, data_type);
      return false;
    }

  return true;
}

bool DatasetV1::preprocess_modwt_inv(size_t const input_index_received,
                                   DATA::TYPE const data_type) {
  if (this->p_n_data <= 1_UZ) {
    ERR(L"No enought data available.");
    return false;
  } else if (this->p_seq_w == 0_UZ) {
    ERR(L"Recurrent depth can not be equal to zero.");
    return false;
  } else if (this->_reference) {
    ERR(L"The dataset is allocate as refence.");
    return false;
  }

  if (data_type == DATA::INPUT) {
    if (this->Xm_coeff == nullptr) {
      ERR(L"`Xm_coeff` is a nullptr.");
      return false;
    }

    if (this->Xm_coeff_size == nullptr) {
      ERR(L"`Xm_coeff_size` is a nullptr.");
      return false;
    }
  } else if (data_type == DATA::OUTPUT) {
    if (this->Ym_coeff == nullptr) {
      ERR(L"`Ym_coeff` is a nullptr.");
      return false;
    }

    if (this->Ym_coeff_size == nullptr) {
      ERR(L"`Ym_coeff_size` is a nullptr.");
      return false;
    }
  } else {
    ERR(L"Type input (%d) is not managed in the function.", data_type);
    return false;
  }

  size_t const tmp_input_size(data_type == DATA::INPUT ? this->p_n_inp
                                                       : this->p_n_out);

  if (input_index_received >= tmp_input_size) {
    ERR(L"Input index (%zu) overflow (%zu).", input_index_received,
        tmp_input_size);
    return false;
  }

  size_t const tmp_coefficient_matrix_size(
      data_type == DATA::INPUT ? this->Xm_coeff_size[input_index_received]
                               : this->Ym_coeff_size[input_index_received]);
  size_t i, t;

  real *tmp_ptr_array_inputs(data_type == DATA::INPUT ? this->X : this->Y),
      *tmp_ptr_array_inputs_inverse;

  if (tmp_ptr_array_inputs == nullptr) {
    ERR(L"`tmp_ptr_array_inputs` is a nullptr.");
    return false;
  }

  tmp_ptr_array_inputs_inverse = new real[this->p_n_data];

  for (t = 0_UZ; t != this->p_seq_w; ++t) {
    real *&tmp_ptr_coefficient_matrix(
        data_type == DATA::INPUT
            ? this->Xm_coeff[input_index_received * this->p_seq_w + t]
            : this->Ym_coeff[input_index_received * this->p_seq_w + t]);

    if (Math::modwt_inverse(tmp_coefficient_matrix_size, this->p_n_data,
                            tmp_ptr_coefficient_matrix,
                            tmp_ptr_array_inputs_inverse) == false) {
      ERR(L"An error has been triggered from the "
          L"`modwt_inverse(%zu, %zu, ptr, ptr)` function.",
          this->p_n_data,
          data_type == DATA::INPUT ? this->Xm_coeff_size[input_index_received]
                                   : this->Ym_coeff_size[input_index_received]);
      return false;
    }

    // Set timed input.
    for (i = 0_UZ; i != this->p_n_data; ++i)
      tmp_ptr_array_inputs[i * tmp_input_size * this->p_seq_w +
                           t * tmp_input_size + input_index_received] =
          tmp_ptr_array_inputs_inverse[i];

    SAFE_DELETE_ARRAY(tmp_ptr_coefficient_matrix)
  }

  delete[](tmp_ptr_array_inputs_inverse);

  if (data_type == DATA::INPUT)
    this->Xm_coeff_size[input_index_received] = 0_UZ;
  else
    this->Ym_coeff_size[input_index_received] = 0_UZ;

  return true;
}

template <typename T>
void shift(size_t const start, size_t const end, size_t const shift,
           T *const data) {
  for (size_t k(start); k != end; --k) data[k + shift] = data[k];
}

template <typename T>
void shift(size_t const row_str, size_t const row_end, size_t const cols,
           size_t const shift, T *const data) {
  for (size_t row(row_str), col; row != row_end; --row)
    for (col = 0_UZ; col != cols; ++col)
      data[(row + shift) * cols + col] = data[row * cols + col];
}

bool DatasetV1::shift_arrays(size_t const input_index_received,
                           size_t const shift_size_received,
                           DATA::TYPE const data_type) {
  size_t const tmp_new_input_size(
      (data_type == DATA::INPUT ? this->p_n_inp : this->p_n_out) +
      shift_size_received);
  size_t i, t, k;

  real *tmp_ptr_array_inputs;

  if (data_type == DATA::INPUT) {
    if (this->Xm_coeff != nullptr) {
      this->Xm_coeff = Mem::reallocate_ptofpt<real *, true>(
          this->Xm_coeff, tmp_new_input_size * this->p_seq_w,
          this->p_n_inp * this->p_seq_w);
      DL::v1::shift<real *>(this->p_n_inp - 1_UZ, input_index_received,
                        this->p_seq_w, shift_size_received, this->Xm_coeff);
      Mem::fill_null<real *>(
          this->Xm_coeff + (input_index_received + 1_UZ) * this->p_seq_w,
          this->Xm_coeff + (input_index_received + 1_UZ + shift_size_received) *
                               this->p_seq_w);
    }

    if (this->Xm_coeff_size != nullptr) {
      this->Xm_coeff_size = Mem::reallocate<size_t, true>(
          this->Xm_coeff_size, tmp_new_input_size, this->p_n_inp);
      DL::v1::shift<size_t>(this->p_n_inp - 1_UZ, input_index_received,
                        shift_size_received, this->Xm_coeff_size);
      memset(this->Xm_coeff_size + (input_index_received + 1_UZ), 0,
             shift_size_received * sizeof(size_t));
    }

    if (this->X != nullptr) {
      tmp_ptr_array_inputs =
          new real[this->p_n_data * this->p_seq_w * tmp_new_input_size];
      memset(
          tmp_ptr_array_inputs, 0,
          this->p_n_data * this->p_seq_w * tmp_new_input_size * sizeof(real));
      for (i = 0_UZ; i != this->p_n_data; ++i) {
        this->Xm[i] =
            tmp_ptr_array_inputs + i * tmp_new_input_size * this->p_seq_w;

        for (t = 0_UZ; t != this->p_seq_w; ++t) {
          // Left inputs.
          for (k = 0_UZ; k != input_index_received + 1_UZ; ++k)
            tmp_ptr_array_inputs[i * tmp_new_input_size * this->p_seq_w +
                                 t * tmp_new_input_size + k] =
                this->X[i * this->p_n_inp * this->p_seq_w + t * this->p_n_inp +
                        k];

          // Right inputs.
          for (k = this->p_n_inp - 1_UZ; k != input_index_received; --k)
            tmp_ptr_array_inputs[i * tmp_new_input_size * this->p_seq_w +
                                 t * tmp_new_input_size + k +
                                 shift_size_received] =
                this->X[i * this->p_n_inp * this->p_seq_w + t * this->p_n_inp +
                        k];
        }
      }
      delete[](this->X);
      this->X = tmp_ptr_array_inputs;
    }

    if (this->_ptr_input_array_scaler__zero_centered != nullptr) {
      this->_ptr_input_array_scaler__zero_centered =
          Mem::reallocate_obj<ScalerZeroCentered, true>(
              this->_ptr_input_array_scaler__zero_centered, tmp_new_input_size,
              this->p_n_inp);
      DL::v1::shift<ScalerZeroCentered>(
          this->p_n_inp - 1_UZ, input_index_received, shift_size_received,
          this->_ptr_input_array_scaler__zero_centered);
      Mem::fill<ScalerZeroCentered>(
          this->_ptr_input_array_scaler__zero_centered +
              (input_index_received + 1_UZ),
          this->_ptr_input_array_scaler__zero_centered +
              (input_index_received + 1_UZ + shift_size_received),
          ScalerZeroCentered());
    }

    if (this->_ptr_input_array_scaler__minimum_maximum != nullptr) {
      this->_ptr_input_array_scaler__minimum_maximum =
          Mem::reallocate_obj<ScalerMinMax, true>(
              this->_ptr_input_array_scaler__minimum_maximum,
              tmp_new_input_size, this->p_n_inp);
      DL::v1::shift<ScalerMinMax>(this->p_n_inp - 1_UZ, input_index_received,
                              shift_size_received,
                              this->_ptr_input_array_scaler__minimum_maximum);
      Mem::fill<ScalerMinMax>(
          this->_ptr_input_array_scaler__minimum_maximum +
              (input_index_received + 1_UZ),
          this->_ptr_input_array_scaler__minimum_maximum +
              (input_index_received + 1_UZ + shift_size_received),
          ScalerMinMax());
    }

    this->p_n_inp += shift_size_received;
  } else if (data_type == DATA::OUTPUT) {
    if (this->Ym_coeff != nullptr) {
      this->Ym_coeff = Mem::reallocate_ptofpt<real *, true>(
          this->Ym_coeff, tmp_new_input_size * this->p_seq_w,
          this->p_n_out * this->p_seq_w);
      DL::v1::shift<real *>(this->p_n_out - 1_UZ, input_index_received,
                        this->p_seq_w, shift_size_received, this->Ym_coeff);
      Mem::fill_null<real *>(
          this->Ym_coeff + (input_index_received + 1_UZ) * this->p_seq_w,
          this->Ym_coeff + (input_index_received + 1_UZ + shift_size_received) *
                               this->p_seq_w);
    }

    if (this->Ym_coeff_size != nullptr) {
      this->Ym_coeff_size = Mem::reallocate<size_t, true>(
          this->Ym_coeff_size, tmp_new_input_size, this->p_n_out);
      DL::v1::shift<size_t>(this->p_n_out - 1_UZ, input_index_received,
                        shift_size_received, this->Ym_coeff_size);
      memset(this->Ym_coeff_size + (input_index_received + 1_UZ), 0,
             shift_size_received * sizeof(size_t));
    }

    if (this->Y != nullptr) {
      tmp_ptr_array_inputs =
          new real[this->p_n_data * this->p_seq_w * tmp_new_input_size];
      memset(
          tmp_ptr_array_inputs, 0,
          this->p_n_data * this->p_seq_w * tmp_new_input_size * sizeof(real));
      for (i = 0_UZ; i != this->p_n_data; ++i) {
        this->Ym[i] =
            tmp_ptr_array_inputs + i * tmp_new_input_size * this->p_seq_w;

        for (t = 0_UZ; t != this->p_seq_w; ++t) {
          // Left inputs.
          for (k = 0_UZ; k != input_index_received + 1_UZ; ++k)
            tmp_ptr_array_inputs[i * tmp_new_input_size * this->p_seq_w +
                                 t * tmp_new_input_size + k] =
                this->Y[i * this->p_n_out * this->p_seq_w + t * this->p_n_out +
                        k];

          // Right inputs.
          for (k = this->p_n_out - 1_UZ; k != input_index_received; --k)
            tmp_ptr_array_inputs[i * tmp_new_input_size * this->p_seq_w +
                                 t * tmp_new_input_size + k +
                                 shift_size_received] =
                this->Y[i * this->p_n_out * this->p_seq_w + t * this->p_n_out +
                        k];
        }
      }
      delete[](this->Y);
      this->Y = tmp_ptr_array_inputs;
    }

    if (this->_ptr_output_array_scaler__zero_centered != nullptr) {
      this->_ptr_output_array_scaler__zero_centered =
          Mem::reallocate_obj<ScalerZeroCentered, true>(
              this->_ptr_output_array_scaler__zero_centered, tmp_new_input_size,
              this->p_n_out);
      DL::v1::shift<ScalerZeroCentered>(
          this->p_n_out - 1_UZ, input_index_received, shift_size_received,
          this->_ptr_output_array_scaler__zero_centered);
      Mem::fill<ScalerZeroCentered>(
          this->_ptr_output_array_scaler__zero_centered +
              (input_index_received + 1_UZ),
          this->_ptr_output_array_scaler__zero_centered +
              (input_index_received + 1_UZ + shift_size_received),
          ScalerZeroCentered());
    }

    if (this->_ptr_output_array_scaler__minimum_maximum != nullptr) {
      this->_ptr_output_array_scaler__minimum_maximum =
          Mem::reallocate_obj<ScalerMinMax, true>(
              this->_ptr_output_array_scaler__minimum_maximum,
              tmp_new_input_size, this->p_n_inp);
      DL::v1::shift<ScalerMinMax>(this->p_n_out - 1_UZ, input_index_received,
                              shift_size_received,
                              this->_ptr_output_array_scaler__minimum_maximum);
      Mem::fill<ScalerMinMax>(
          this->_ptr_output_array_scaler__minimum_maximum +
              (input_index_received + 1_UZ),
          this->_ptr_output_array_scaler__minimum_maximum +
              (input_index_received + 1_UZ + shift_size_received),
          ScalerMinMax());
    }

    this->p_n_out += shift_size_received;
  } else {
    ERR(L"Type input (%d) is not managed in the function.", data_type);
    return false;
  }

  return true;
}

bool DatasetV1::Preprocessing__Merge__MODWT(size_t const desired_J_level_received,
                                          DATA::TYPE const data_type) {
  if (this->p_n_data <= 1_UZ) {
    ERR(L"No enought data available.");
    return false;
  } else if (this->p_seq_w == 0_UZ) {
    ERR(L"Recurrent depth can not be equal to zero.");
    return false;
  } else if (this->_reference) {
    ERR(L"The dataset is allocate as refence.");
    return false;
  } else if (desired_J_level_received == 0_UZ) {
    return true;
  }

  size_t const tmp_input_size(data_type == DATA::INPUT ? this->p_n_inp
                                                       : this->p_n_out);
  size_t k, tmp_shift_index;

  for (tmp_shift_index = 0_UZ, k = 0_UZ; k != tmp_input_size; ++k) {
    if (this->Preprocessing__Merge__MODWT(k + tmp_shift_index,
                                          desired_J_level_received,
                                          data_type) == false) {
      ERR(L"An error has been triggered from the "
          L"`Preprocessing__Merge__MODWT(%zu, %zu, %d)` function.",
          k + tmp_shift_index, desired_J_level_received, data_type);
      return false;
    }

    tmp_shift_index += desired_J_level_received;
  }

  return true;
}

bool DatasetV1::Preprocessing__Merge__MODWT(size_t const input_index_received,
                                          size_t const desired_J_level_received,
                                          DATA::TYPE const data_type) {
  // Valid dataset.
  if (this->p_n_data <= 1_UZ) {
    ERR(L"No enought data available.");
    return false;
  } else if (this->p_seq_w == 0_UZ) {
    ERR(L"Recurrent depth can not be equal to zero.");
    return false;
  } else if (this->_reference) {
    ERR(L"The dataset is allocate as refence.");
    return false;
  } else if (desired_J_level_received == 0_UZ) {
    return true;
  }
  // |END| Valid dataset. |END|

  // Variables.
  size_t const tmp_J_level(std::min<size_t>(this->MODWT__J_Level_Maximum(),
                                            desired_J_level_received));
  size_t tmp_coefficient_matrix_size, i, t, tmp_j_index;
  // |END| Variables. |END|

  // Valid input index.
  size_t const tmp_new_input_size(
      (data_type == DATA::INPUT ? this->p_n_inp : this->p_n_out) + tmp_J_level);

  if (input_index_received >= tmp_new_input_size - tmp_J_level) {
    ERR(L"Input index (%zu) overflow (%zu).", input_index_received,
        tmp_new_input_size - tmp_J_level);
    return false;
  }
  // |END| Valid input index. |END|

  // allocate || reallocate.
  if (data_type == DATA::INPUT) {
    if (this->shift_arrays(input_index_received, tmp_J_level, DATA::INPUT) ==
        false) {
      ERR(L"An error has been triggered from the "
          L"`shift_arrays(%zu, %zu, %d)` function.",
          input_index_received, tmp_J_level, DATA::INPUT);
      return false;
    }

    // allocate.
    if (this->Xm_coeff == nullptr) {
      this->Xm_coeff = new real *[tmp_new_input_size * this->p_seq_w];
      Mem::fill_null<real *>(
          this->Xm_coeff, this->Xm_coeff + tmp_new_input_size * this->p_seq_w);
    }

    // allocate.
    if (this->Xm_coeff_size == nullptr) {
      this->Xm_coeff_size = new size_t[tmp_new_input_size];
      memset(this->Xm_coeff_size, 0, tmp_new_input_size * sizeof(size_t));
    }
  } else if (data_type == DATA::OUTPUT) {
    if (this->shift_arrays(input_index_received, tmp_J_level, DATA::OUTPUT) ==
        false) {
      ERR(L"An error has been triggered from the "
          L"`shift_arrays(%zu, %zu, %d)` function.",
          input_index_received, tmp_J_level, DATA::OUTPUT);
      return false;
    }

    // allocate.
    if (this->Ym_coeff == nullptr) {
      this->Ym_coeff = new real *[tmp_new_input_size * this->p_seq_w];
      Mem::fill_null<real *>(
          this->Ym_coeff, this->Ym_coeff + tmp_new_input_size * this->p_seq_w);
    }

    // allocate.
    if (this->Ym_coeff_size == nullptr) {
      this->Ym_coeff_size = new size_t[tmp_new_input_size];
      memset(this->Ym_coeff_size, 0, tmp_new_input_size * sizeof(size_t));
    }
  } else {
    ERR(L"Type input (%d) is not managed in the function.", data_type);
    return false;
  }
  // |END| allocate || reallocate. |END|

  // allocate tempory inputs storage.
  real *const tmp_ptr_array_inputs(data_type == DATA::INPUT ? this->X
                                                            : this->Y),
      *tmp_ptr_array_inputs_preproced;

  if (tmp_ptr_array_inputs == nullptr) {
    ERR(L"`tmp_ptr_array_inputs` is a nullptr.");
    return false;
  }

  tmp_ptr_array_inputs_preproced = new real[this->p_n_data];
  // |END| allocate tempory inputs storage. |END|

  for (t = 0_UZ; t != this->p_seq_w; ++t) {
    real *&tmp_ptr_coefficient_matrix(
        data_type == DATA::INPUT
            ? this->Xm_coeff[input_index_received * this->p_seq_w + t]
            : this->Ym_coeff[input_index_received * this->p_seq_w + t]);

    tmp_coefficient_matrix_size =
        data_type == DATA::INPUT ? this->Xm_coeff_size[input_index_received]
                                 : this->Ym_coeff_size[input_index_received];

    // Get timed input.
    for (i = 0_UZ; i != this->p_n_data; ++i)
      tmp_ptr_array_inputs_preproced[i] =
          tmp_ptr_array_inputs[i * tmp_new_input_size * this->p_seq_w +
                               t * tmp_new_input_size + input_index_received];

    if (Math::modwt(this->p_n_data, tmp_coefficient_matrix_size,
                    tmp_ptr_array_inputs_preproced, tmp_ptr_coefficient_matrix,
                    tmp_J_level) == false) {
      ERR(L"An error has been triggered from the "
          L"`modwt(%zu, %zu, ptr, ptr, %zu)` function.",
          this->p_n_data, tmp_coefficient_matrix_size, tmp_J_level);
      return false;
    }

    // Set timed input.
    for (tmp_j_index = 0_UZ; tmp_j_index != tmp_J_level + 1_UZ; ++tmp_j_index)
      for (i = 0_UZ; i != this->p_n_data; ++i)
        tmp_ptr_array_inputs[i * tmp_new_input_size * this->p_seq_w +
                             t * tmp_new_input_size + input_index_received +
                             tmp_j_index] =
            tmp_ptr_coefficient_matrix[tmp_j_index * this->p_n_data + i];
  }

  // Delete tempory inputs storage.
  delete[](tmp_ptr_array_inputs_preproced);
  // |END| Delete tempory inputs storage. |END|

  // Store the matrix size.
  if (data_type == DATA::INPUT)
    this->Xm_coeff_size[input_index_received] = tmp_coefficient_matrix_size;
  else
    this->Ym_coeff_size[input_index_received] = tmp_coefficient_matrix_size;
  // |END| Store the matrix size. |END|

  this->Compute__Start_Index();

  return true;
}

bool DatasetV1::Preprocessing__Merge__MODWT(size_t const input_index_received,
                                          size_t const input_size_received,
                                          real *&ptr_array_inputs_received,
                                          DATA::TYPE const data_type) {
  // Valid dataset.
  if (this->p_n_data <= 1_UZ) {
    ERR(L"No enought data available.");
    return false;
  } else if (this->p_seq_w == 0_UZ) {
    ERR(L"Recurrent depth can not be equal to zero.");
    return false;
  } else if (this->_reference) {
    ERR(L"The dataset is allocate as refence.");
    return false;
  }

  if (data_type == DATA::INPUT) {
    if (this->Xm_coeff == nullptr) {
      ERR(L"`Xm_coeff` is a nullptr.");
      return false;
    }

    if (this->Xm_coeff_size == nullptr) {
      ERR(L"`Xm_coeff_size` is a nullptr.");
      return false;
    }
  } else if (data_type == DATA::OUTPUT) {
    if (this->Ym_coeff == nullptr) {
      ERR(L"`Ym_coeff` is a nullptr.");
      return false;
    }

    if (this->Ym_coeff_size == nullptr) {
      ERR(L"`Ym_coeff_size` is a nullptr.");
      return false;
    }
  } else {
    ERR(L"Type input (%d) is not managed in the function.", data_type);
    return false;
  }
  // |END| Valid dataset. |END|

  // Variables.
  size_t const tmp_batch_size(this->p_n_data + 1_UZ),
      tmp_input_coefficient_matrix_size(
          data_type == DATA::INPUT ? this->Xm_coeff_size[input_index_received]
                                   : this->Ym_coeff_size[input_index_received]);
  size_t tmp_J_level(tmp_input_coefficient_matrix_size / this->p_n_data),
      tmp_coefficient_matrix_size, i, t, k, tmp_j_index;
  // |END| Variables. |END|

  // Valid input index.
  if (tmp_J_level == 0_UZ)
    return true;
  else
    --tmp_J_level;

  size_t const tmp_input_size(data_type == DATA::INPUT ? this->p_n_inp
                                                       : this->p_n_out);

  if (input_index_received >= tmp_input_size) {
    ERR(L"Input index (%zu) overflow (%zu).", input_index_received,
        tmp_input_size);
    return false;
  } else if (input_index_received >= input_size_received) {
    ERR(L"Input index (%zu) overflow (%zu).", input_index_received,
        input_size_received);
    return false;
  }
  // |END| Valid input index. |END|

  // reallocate.
  size_t const tmp_new_input_size(input_size_received + tmp_J_level);

  real *tmp_ptr_array_inputs(new real[this->p_seq_w * tmp_new_input_size]);

  for (t = 0_UZ; t != this->p_seq_w; ++t) {
    // Left inputs.
    for (k = 0_UZ; k != input_index_received + 1_UZ; ++k)
      tmp_ptr_array_inputs[t * tmp_new_input_size + k] =
          ptr_array_inputs_received[t * input_size_received + k];

    // Right inputs.
    for (k = input_size_received - 1_UZ; k != input_index_received; --k)
      tmp_ptr_array_inputs[t * tmp_new_input_size + k + tmp_J_level] =
          ptr_array_inputs_received[t * input_size_received + k];
  }

  delete[](ptr_array_inputs_received);
  ptr_array_inputs_received = tmp_ptr_array_inputs;
  // |END| reallocate. |END|

  // allocate tempory inputs storage.
  real const *tmp_ptr_source_coefficient_matrix;
  real *tmp_ptr_array_inputs_preproced(new real[tmp_batch_size]),
      *tmp_ptr_coefficient_matrix;
  // |END| allocate tempory inputs storage. |END|

  for (t = 0_UZ; t != this->p_seq_w; ++t) {
    tmp_ptr_source_coefficient_matrix =
        (data_type == DATA::INPUT
             ? this->Xm_coeff[input_index_received * this->p_seq_w + t]
             : this->Ym_coeff[input_index_received * this->p_seq_w + t]);
    tmp_ptr_coefficient_matrix = nullptr;

    tmp_coefficient_matrix_size = 0_UZ;

    // Get timed input from dataset.
    for (i = 0_UZ; i != this->p_n_data; ++i) {
      tmp_ptr_array_inputs_preproced[i] = tmp_ptr_source_coefficient_matrix[i];
    }
    for (tmp_j_index = 1_UZ; tmp_j_index != tmp_J_level + 1_UZ; ++tmp_j_index) {
      for (i = 0_UZ; i != this->p_n_data; ++i) {
        tmp_ptr_array_inputs_preproced[i] +=
            tmp_ptr_source_coefficient_matrix[tmp_j_index * this->p_n_data + i];
      }
    }
    // |END| Get timed input from dataset. |END|

    // Get timed input from arguments.
    tmp_ptr_array_inputs_preproced[i] =
        tmp_ptr_array_inputs[t * tmp_new_input_size + input_index_received];

    if (Math::modwt(tmp_batch_size, tmp_coefficient_matrix_size,
                    tmp_ptr_array_inputs_preproced, tmp_ptr_coefficient_matrix,
                    tmp_J_level) == false) {
      ERR(L"An error has been triggered from the "
          L"`modwt(%zu, %zu, ptr, ptr, %zu)` function.",
          tmp_batch_size, tmp_coefficient_matrix_size, tmp_J_level);
      return false;
    }

    // Set timed input into arguments.
    for (tmp_j_index = 0_UZ; tmp_j_index != tmp_J_level + 1_UZ; ++tmp_j_index) {
      tmp_ptr_array_inputs[t * tmp_new_input_size + input_index_received +
                           tmp_j_index] =
          tmp_ptr_coefficient_matrix[tmp_j_index * tmp_batch_size + i];
    }

    delete[](tmp_ptr_coefficient_matrix);
  }

  // Delete tempory inputs storage.
  delete[](tmp_ptr_array_inputs_preproced);
  // |END| Delete tempory inputs storage. |END|

  return true;
}

bool DatasetV1::Preprocessing__Sequence_Window(
    size_t const sequence_window_received,
    size_t const sequence_horizon_received, real *&ptr_array_inputs_received) {
  if (sequence_window_received <= 1_UZ)
    return true;
  else if (sequence_horizon_received == 0_UZ)
    return true;

  real const *const tmp_ptr_array_previous_inputs(
      this->X + this->p_n_data * this->p_n_inp -
      ((sequence_window_received - 1_UZ) + (sequence_horizon_received - 1_UZ)) *
          this->p_n_inp);
  real *tmp_ptr_array_inputs(
      new real[sequence_window_received * this->p_n_inp]);

  for (size_t tmp_index(0_UZ);
       tmp_index != (sequence_window_received - 1_UZ) * this->p_n_inp;
       ++tmp_index) {
    tmp_ptr_array_inputs[tmp_index] = tmp_ptr_array_previous_inputs[tmp_index];
  }

  memcpy(
      tmp_ptr_array_inputs + (sequence_window_received - 1_UZ) * this->p_n_inp,
      ptr_array_inputs_received, this->p_n_inp * sizeof(real));

  delete[](ptr_array_inputs_received);

  ptr_array_inputs_received = tmp_ptr_array_inputs;

  return true;
}

bool DatasetV1::valide_spec(size_t const number_inputs_received,
                          size_t const number_outputs_received,
                          size_t const number_recurrent_depth_received) const {
  if (this->p_n_inp != number_inputs_received) {
    ERR(L"The number of inputs (%zu) differ from the number of inputs "
        L"received as argument (%zu).",
        this->p_n_inp, number_inputs_received);
    return false;
  } else if (this->p_n_out != number_outputs_received) {
    ERR(L"The number of outputs (%zu) differ from the number of outputs "
        L"received as argument (%zu).",
        this->p_n_out, number_outputs_received);
    return false;
  } else if (this->p_seq_w != number_recurrent_depth_received) {
    ERR(L"The number of recurrent depth (%zu) differ from the number of "
        L"recurrent depth received as argument (%zu).",
        this->p_seq_w, number_recurrent_depth_received);
    return false;
  }

  return true;
}

bool DatasetV1::Get__Reference(void) const { return (this->_reference); }

bool DatasetV1::Use__Multi_Label(void) const { return (this->_use_multi_label); }

bool DatasetV1::Deallocate(void) {
  if (this->_reference) {
    this->X = nullptr;
    this->Xm = nullptr;

    this->Y = nullptr;
    this->Ym = nullptr;

    this->Xm_coeff = nullptr;
    this->Ym_coeff = nullptr;

    this->Xm_coeff_size = nullptr;
    this->Ym_coeff_size = nullptr;

    this->_ptr_input_array_scaler__minimum_maximum = nullptr;
    this->_ptr_output_array_scaler__minimum_maximum = nullptr;

    this->_ptr_input_array_scaler__zero_centered = nullptr;
    this->_ptr_output_array_scaler__zero_centered = nullptr;
  } else {
    size_t k;

    SAFE_DELETE_ARRAY(this->X);
    SAFE_DELETE_ARRAY(this->Xm);

    SAFE_DELETE_ARRAY(this->Y);
    SAFE_DELETE_ARRAY(this->Ym);

    if (this->Xm_coeff != nullptr) {
      for (k = 0_UZ; k != this->p_n_inp * this->p_seq_w; ++k) {
        SAFE_DELETE_ARRAY(this->Xm_coeff[k]);
      }

      SAFE_DELETE_ARRAY(this->Xm_coeff);
    }

    if (this->Ym_coeff != nullptr) {
      for (k = 0_UZ; k != this->p_n_out * this->p_seq_w; ++k) {
        SAFE_DELETE_ARRAY(this->Ym_coeff[k]);
      }

      SAFE_DELETE_ARRAY(this->Ym_coeff);
    }

    SAFE_DELETE_ARRAY(this->Xm_coeff_size);
    SAFE_DELETE_ARRAY(this->Ym_coeff_size);

    SAFE_DELETE_ARRAY(this->_ptr_input_array_scaler__minimum_maximum);
    SAFE_DELETE_ARRAY(this->_ptr_output_array_scaler__minimum_maximum);

    SAFE_DELETE_ARRAY(this->_ptr_input_array_scaler__zero_centered);
    SAFE_DELETE_ARRAY(this->_ptr_output_array_scaler__zero_centered);
  }

  return true;
}

size_t DatasetV1::get_n_data(void) const {
  return (this->p_n_data - this->p_str_i);
}

size_t DatasetV1::get_n_batch(void) const { return (1_UZ); }

size_t DatasetV1::get_n_inp(void) const { return (this->p_n_inp); }

size_t DatasetV1::get_n_out(void) const { return (this->p_n_out); }

size_t DatasetV1::get_seq_w(void) const { return (this->p_seq_w); }

size_t DatasetV1::MODWT__J_Level_Maximum(void) const {
  return (static_cast<size_t>(
      floor(log(static_cast<double>(this->p_n_data)) / log(2.0))));
}

double DatasetV1::train(Model *const model) {
  if (this->valide_spec(model->n_inp, model->n_out, model->seq_w) == false) {
    ERR(L"An error has been triggered from the "
        L"`valide_spec(%zu, %zu, %zu)` function.",
        model->n_inp, model->n_out, model->seq_w);
    return HUGE_VAL;
  }

  // Initialize training.
  if (model->ptr_array_derivatives_parameters == nullptr)
    model->clear_training_arrays();

  model->type_state_propagation = PROPAGATION::TRAINING;

  if (model->Set__Multi_Label(this->Use__Multi_Label()) == false) {
    ERR(L"An error has been triggered from the "
        L"`Set__Multi_Label(%ls)` function.",
        this->Use__Multi_Label() ? "true" : "false");
    return HUGE_VAL;
  } else if (model->update_mem_batch_size(this->get_n_data()) == false) {
    ERR(L"An error has been triggered from the "
        L"`update_mem_batch_size(%zu)` function.",
        this->get_n_data());
    return HUGE_VAL;
  } else if (model->weights_initialized() == false &&
             model->initialize_weights(this) == false) {
    ERR(L"An error has been triggered from the "
        L"`initialize_weights(self)` function.");
    return HUGE_VAL;
  }

  double loss;

  if (model->use_mp && model->is_mp_initialized) {
    if (model->update_mem_thread_size(this->get_n_data()) == false) {
      ERR(L"An error has been triggered from the "
          L"`update_mem_thread_size(%zu)` function.",
          this->get_n_data());
      return (std::numeric_limits<real>::max)();
    }

    omp_set_num_threads(static_cast<int>(model->number_threads));

    loss = this->train_mp(model);
  } else {
    loss = this->train_st(model);
  }

  model->type_state_propagation = PROPAGATION::INFERENCE;

  return loss;
}

double DatasetV1::train_mp(Model *const model) {
  this->Train_Epoch_OpenMP(model);

  model->update_weights_mp(this->get_n_data(), this->get_n_data());

  model->epoch_time_step += 1_r;

  ENV::TYPE const def_env_type(this->env_type == ENV::NONE ? ENV::TRAIN
                                                           : this->env_type);
  model->set_loss(def_env_type, model->get_loss(ENV::NONE));
  model->set_accu(
      def_env_type,
      this->compute_accuracy(this->get_n_data(), this->Get__Input_Array(),
                             this->Get__Output_Array(), model));

  return model->get_loss(def_env_type);
}

double DatasetV1::train_st(Model *const model) {
  this->Train_Epoch_Loop(model);

  model->update_weights_st(this->get_n_data(), this->get_n_data());

  model->epoch_time_step += 1_r;

  ENV::TYPE const def_env_type(this->env_type == ENV::NONE ? ENV::TRAIN
                                                           : this->env_type);
  model->set_loss(def_env_type, model->get_loss(ENV::NONE));
  model->set_accu(
      def_env_type,
      this->compute_accuracy(this->get_n_data(), this->Get__Input_Array(),
                             this->Get__Output_Array(), model));

  return model->get_loss(def_env_type);
}

double DatasetV1::compute_accuracy(
    size_t const batch_size, real const *const *const ptr_array_inputs_received,
    real const *const *const ptr_array_desired_outputs_received,
    Model *const model) {
  if (this->valide_spec(model->n_inp, model->n_out, model->seq_w) == false) {
    ERR(L"An error has been triggered from the "
        L"`valide_spec(%zu, %zu, %zu)` function.",
        model->n_inp, model->n_out, model->seq_w);
    return HUGE_VAL;
  }

  if (model->type_accuracy_function == ACCU_FN::R) {
    size_t const tmp_output_size(model->get_n_out()),
        tmp_maximum_batch_size(model->batch_size),
        tmp_number_batchs(static_cast<size_t>(
            ceil(static_cast<double>(batch_size) /
                 static_cast<double>(tmp_maximum_batch_size))));
    size_t tmp_batch_index(0_UZ), tmp_batch_size(0_UZ);

    // Mean.
    *model->ptr_array_accuracy_values[0] /= static_cast<double>(
        batch_size * (this->get_seq_w() - model->n_time_delay) *
        tmp_output_size);
    *model->ptr_array_accuracy_values[1] /= static_cast<double>(
        batch_size * (this->get_seq_w() - model->n_time_delay) *
        tmp_output_size);

    if (model->use_mp && model->is_mp_initialized) {
      omp_set_num_threads(static_cast<int>(model->number_threads));

#pragma omp parallel private(tmp_batch_index, tmp_batch_size)
      for (tmp_batch_index = 0_UZ; tmp_batch_index != tmp_number_batchs;
           ++tmp_batch_index) {
        tmp_batch_size =
            tmp_batch_index + 1_UZ != tmp_number_batchs
                ? tmp_maximum_batch_size
                : batch_size - tmp_batch_index * tmp_maximum_batch_size;

        model->forward_pass(tmp_batch_size,
                            ptr_array_inputs_received +
                                tmp_batch_index * tmp_maximum_batch_size);

        model->compute_r(tmp_batch_size,
                         ptr_array_desired_outputs_received +
                             tmp_batch_index * tmp_maximum_batch_size);
      }

      model->Merge__Accuracy__R();
    } else {
      for (tmp_batch_index = 0_UZ; tmp_batch_index != tmp_number_batchs;
           ++tmp_batch_index) {
        tmp_batch_size =
            tmp_batch_index + 1_UZ != tmp_number_batchs
                ? tmp_maximum_batch_size
                : batch_size - tmp_batch_index * tmp_maximum_batch_size;

        model->forward_pass(tmp_batch_size,
                            ptr_array_inputs_received +
                                tmp_batch_index * tmp_maximum_batch_size);

        model->compute_r(tmp_batch_size,
                         ptr_array_desired_outputs_received +
                             tmp_batch_index * tmp_maximum_batch_size);
      }
    }

    // R = numerator / (sqrt(denominator_desired) *
    // sqrt(denominator_predicted)).
    *model->ptr_array_accuracy_values[0] =
        *model->ptr_array_accuracy_values[3] *
                    *model->ptr_array_accuracy_values[4] ==
                0.0
            ? 0.0
            : *model->ptr_array_accuracy_values[2] /
                  (sqrt(*model->ptr_array_accuracy_values[3]) *
                   sqrt(*model->ptr_array_accuracy_values[4]));
  }

  return model->get_accu(ENV::NONE);
}

double DatasetV1::evaluate(Model *const model) {
  if (this->valide_spec(model->n_inp, model->n_out, model->seq_w) == false) {
    ERR(L"An error has been triggered from the "
        L"`valide_spec(%zu, %zu, %zu)` function.",
        model->n_inp, model->n_out, model->seq_w);
    return HUGE_VAL;
  }

  model->reset_loss();

  if (model->Set__Multi_Label(this->Use__Multi_Label()) == false) {
    ERR(L"An error has been triggered from the "
        L"`Set__Multi_Label(%ls)` function.",
        to_wstring(this->Use__Multi_Label()).c_str());
    return HUGE_VAL;
  } else if (model->update_mem_batch_size(this->DatasetV1::get_n_data()) == false) {
    ERR(L"An error has been triggered from the "
        L"`update_mem_batch_size(%zu)` function.",
        this->DatasetV1::get_n_data());
    return HUGE_VAL;
  } else if (model->weights_initialized() == false &&
             model->initialize_weights(this) == false) {
    ERR(L"An error has been triggered from the "
        L"`initialize_weights(self)` function.");
    return HUGE_VAL;
  }

  if (model->use_mp && model->is_mp_initialized) {
    if (model->update_mem_thread_size(this->DatasetV1::get_n_data()) == false) {
      ERR(L"An error has been triggered from the "
          L"`update_mem_thread_size(%zu)` function.",
          this->DatasetV1::get_n_data());
      return HUGE_VAL;
    }

    omp_set_num_threads(static_cast<int>(model->number_threads));

    return this->evaluate_mp(model);
  } else {
    return this->evaluate_st(model);
  }
}

double DatasetV1::evaluate_mp(Model *const model) {
  size_t const n_data(this->DatasetV1::get_n_data()),
      tmp_maximum_batch_size(model->batch_size),
      tmp_number_batchs(static_cast<size_t>(
          ceil(static_cast<double>(n_data) /
               static_cast<double>(tmp_maximum_batch_size))));
  size_t tmp_batch_index(0_UZ), tmp_batch_size(0_UZ);

#pragma omp parallel private(tmp_batch_index, tmp_batch_size)
  for (tmp_batch_index = 0_UZ; tmp_batch_index != tmp_number_batchs;
       ++tmp_batch_index) {
    tmp_batch_size = tmp_batch_index + 1_UZ != tmp_number_batchs
                         ? tmp_maximum_batch_size
                         : n_data - tmp_batch_index * tmp_maximum_batch_size;

    model->forward_pass(tmp_batch_size,
                        this->DatasetV1::Get__Input_Array() +
                            tmp_batch_index * tmp_maximum_batch_size);

    model->compute_loss(tmp_batch_size,
                        this->DatasetV1::Get__Output_Array() +
                            tmp_batch_index * tmp_maximum_batch_size);
  }

  model->merge_mp_accu_loss();

  model->n_acc_trial = n_data * (this->get_seq_w() - model->n_time_delay) *
                       (model->type_accuracy_function == ACCU_FN::CROSS_ENTROPY
                            ? 1_UZ
                            : model->get_n_out());

  ENV::TYPE const def_env_type(this->env_type == ENV::NONE ? ENV::TESTG
                                                           : this->env_type);
  model->set_loss(def_env_type, model->get_loss(ENV::NONE));
  model->set_accu(
      def_env_type,
      this->compute_accuracy(n_data, this->DatasetV1::Get__Input_Array(),
                             this->DatasetV1::Get__Output_Array(), model));

  return (model->get_loss(def_env_type));
}

#if DEEPLEARNING_USE_ADEPT
void set_gradient_to_output(
    size_t const batch_size, size_t const total_batch_size_received,
    size_t const number_time_delay_received,
    size_t const number_recurrent_depth_received,
    size_t const number_outputs_received,
    real const *const *const ptr_array_outputs_array_received,
    var *const ptr_array_outputs_received) {
  size_t i, k, t;

  real target, diff;
  var *predicted;

  for (i = 0_UZ; i != batch_size; ++i) {
    for (t = number_time_delay_received; t != number_recurrent_depth_received;
         ++t) {
      INFO(L"O-DATA[%zu], TIME[%zu]:", i, t);
      for (k = 0_UZ; k != number_outputs_received; ++k) {
        predicted =
            &ptr_array_outputs_received[i * number_outputs_received +
                                        total_batch_size_received *
                                            number_outputs_received * t +
                                        k];

        target =
            ptr_array_outputs_array_received[i]
                                            [t * number_outputs_received + k];

        diff = cast(*predicted) - target;

        INFO(L"%f ", diff);

        predicted->set_gradient(diff);
      }
      INFO(L"");
    }
  }
}
#endif

void DatasetV1::Adept__Gradient(Model *const model) {
  size_t tmp_batch_size(this->DatasetV1::get_n_data());
  // size_t tmp_batch_size(std::min<size_t>(4_UZ, this->DatasetV1::get_n_data());

  if (model->update_mem_batch_size(tmp_batch_size) == false) {
    ERR(L"An error has been triggered from the "
        L"`update_mem_batch_size(%zu)` function.",
        tmp_batch_size);
    return;
  } else if (model->weights_initialized() == false &&
             model->initialize_weights(this) == false) {
    ERR(L"An error has been triggered from the "
        L"`initialize_weights(self)` function.");
    return;
  }

  model->type_state_propagation = PROPAGATION::TRAINING;

#if DEEPLEARNING_USE_ADEPT
  adept::active_stack()->new_recording();
#endif

  if (model->use_mp && model->is_mp_initialized) {
    if (model->update_mem_thread_size(tmp_batch_size) == false) {
      ERR(L"An error has been triggered from the "
          L"`update_mem_thread_size(%zu)` function.",
          tmp_batch_size);
      return;
    }

    omp_set_num_threads(static_cast<int>(model->number_threads));

#pragma omp parallel
    model->forward_pass(tmp_batch_size, this->DatasetV1::Get__Input_Array());
  } else {
    model->forward_pass(tmp_batch_size, this->DatasetV1::Get__Input_Array());
  }

#if DEEPLEARNING_USE_ADEPT
  set_gradient_to_output(
      tmp_batch_size, model->batch_size, model->n_time_delay, model->seq_w,
      *(model->ptr_last_layer - 1)->ptr_number_outputs, this->Ym,
      (model->ptr_last_layer - 1)->ptr_array_outputs);

  adept::active_stack()->reverse();
#endif

  if (model->use_mp && model->is_mp_initialized) {
#pragma omp parallel
    {
      model->compute_error(tmp_batch_size, this->DatasetV1::Get__Output_Array());

      model->backward_pass(tmp_batch_size);

      model->update_derivatives(
          tmp_batch_size, model->ptr_array_layers + 1, model->ptr_last_layer);
    }

    model->merge_mp_derivatives(0_UZ, model->total_parameters);
  } else {
    model->compute_error(tmp_batch_size, this->DatasetV1::Get__Output_Array());

    model->backward_pass(tmp_batch_size);

    model->update_derivatives(tmp_batch_size, model->ptr_array_layers + 1,
                                    model->ptr_last_layer);
  }

#if DEEPLEARNING_USE_ADEPT
  INFO(L"Autodiff dx/dw: ");
  for (size_t w(0_UZ); w != model->total_parameters; ++w) {
    std::wcout << to_wstring(model->ptr_array_parameters[w].get_gradient(), 8);
    if (w != 0_UZ && (w + 1) % 20 == 0_UZ) INFO(L"");
  }
  INFO(L"");
#endif

  INFO(L"Handcoded dx/dw: ");
  for (size_t w(0_UZ); w != model->total_parameters; ++w) {
    std::wcout << to_wstring(model->ptr_array_derivatives_parameters[w], 8);
    if (w != 0_UZ && (w + 1) % 20 == 0_UZ) INFO(L"");
  }
  INFO(L"");

  model->type_state_propagation = PROPAGATION::INFERENCE;

  ENV::TYPE const def_env_type(this->env_type == ENV::NONE ? ENV::TESTG
                                                           : this->env_type);
  model->set_loss(def_env_type, model->get_loss(ENV::NONE));
  model->set_accu(
      def_env_type,
      this->compute_accuracy(tmp_batch_size, this->Get__Input_Array(),
                             this->Get__Output_Array(), model));
}

double DatasetV1::evaluate_st(Model *const model) {
  size_t const n_data(this->DatasetV1::get_n_data()),
      tmp_maximum_batch_size(model->batch_size),
      tmp_number_batchs(static_cast<size_t>(
          ceil(static_cast<double>(n_data) /
               static_cast<double>(tmp_maximum_batch_size))));
  size_t tmp_batch_size, tmp_batch_index;

  for (tmp_batch_index = 0_UZ; tmp_batch_index != tmp_number_batchs;
       ++tmp_batch_index) {
    tmp_batch_size = tmp_batch_index + 1_UZ != tmp_number_batchs
                         ? tmp_maximum_batch_size
                         : n_data - tmp_batch_index * tmp_maximum_batch_size;

    model->forward_pass(tmp_batch_size,
                        this->DatasetV1::Get__Input_Array() +
                            tmp_batch_index * tmp_maximum_batch_size);

    model->compute_loss(tmp_batch_size,
                        this->DatasetV1::Get__Output_Array() +
                            tmp_batch_index * tmp_maximum_batch_size);
  }

  model->n_acc_trial = n_data * (this->get_seq_w() - model->n_time_delay) *
                       (model->type_accuracy_function == ACCU_FN::CROSS_ENTROPY
                            ? 1_UZ
                            : model->get_n_out());

  ENV::TYPE const def_env_type(this->env_type == ENV::NONE ? ENV::TESTG
                                                           : this->env_type);
  model->set_loss(def_env_type, model->get_loss(ENV::NONE));
  model->set_accu(
      def_env_type,
      this->compute_accuracy(n_data, this->DatasetV1::Get__Input_Array(),
                             this->DatasetV1::Get__Output_Array(), model));

  return model->get_loss(def_env_type);
}

DATASET::TYPE DatasetV1::Get__Type_Dataset_Process(void) const {
  return (this->p_type_dataset_process);
}

real DatasetV1::get_min(size_t const data_start_index_received,
                      size_t const data_end_index_received,
                      size_t const input_index_received,
                      DATA::TYPE const data_type) const {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");
    return 0_r;
  } else if (data_start_index_received > data_end_index_received) {
    ERR(L"Start index (%zu) can not be greater than end index (%zu).",
        data_start_index_received, data_end_index_received);
    return 0_r;
  } else if (data_end_index_received > this->DatasetV1::get_n_data()) {
    ERR(L"End index (%zu) can not be greater than total examples (%zu).",
        data_end_index_received, this->DatasetV1::get_n_data());
    return 0_r;
  } else if (data_type > DATA::OUTPUT) {
    ERR(L"Type input (%d) is not managed in the function.", data_type);
    return 0_r;
  }

  size_t const tmp_input_size(data_type == DATA::INPUT ? this->p_n_inp
                                                       : this->p_n_out);
  size_t i, t;

  if (input_index_received >= tmp_input_size) {
    ERR(L"Input index (%zu) overflow (%zu).", input_index_received,
        tmp_input_size);
    return 0_r;
  }

  real const *const *const tmp_ptr_array_inputs(
      data_type == DATA::INPUT ? this->DatasetV1::Get__Input_Array()
                               : this->DatasetV1::Get__Output_Array());

  if (tmp_ptr_array_inputs == nullptr) {
    ERR(L"`tmp_ptr_array_inputs` is a nullptr.");
    return 0_r;
  }

  real tmp_minimum_value((std::numeric_limits<real>::max)()), tmp_input_value;

  for (i = data_start_index_received; i != data_end_index_received; ++i) {
    for (t = 0_UZ; t != this->p_seq_w; ++t) {
      tmp_input_value =
          tmp_ptr_array_inputs[i][t * tmp_input_size + input_index_received];

      if (tmp_input_value < tmp_minimum_value) {
        tmp_minimum_value = tmp_input_value;
      }
    }
  }

  return (tmp_minimum_value);
}

real DatasetV1::get_min(size_t const data_start_index_received,
                      size_t const data_end_index_received,
                      DATA::TYPE const data_type) const {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");
    return 0_r;
  } else if (data_start_index_received > data_end_index_received) {
    ERR(L"Start index (%zu) can not be greater than end index (%zu).",
        data_start_index_received, data_end_index_received);
    return 0_r;
  } else if (data_end_index_received > this->DatasetV1::get_n_data()) {
    ERR(L"End index (%zu) can not be greater than total examples (%zu).",
        data_end_index_received, this->DatasetV1::get_n_data());
    return 0_r;
  } else if (data_type > DATA::OUTPUT) {
    ERR(L"Type input (%d) is not managed in the function.", data_type);
    return 0_r;
  }

  size_t const tmp_input_size(data_type == DATA::INPUT ? this->p_n_inp
                                                       : this->p_n_out);
  size_t k;

  real tmp_value,
      tmp_minimum_value(this->get_min(
          data_start_index_received, data_end_index_received, 0_UZ, data_type));

  for (k = 1_UZ; k != tmp_input_size; ++k) {
    tmp_value = this->get_min(data_start_index_received,
                              data_end_index_received, k, data_type);

    if (tmp_value < tmp_minimum_value)
      tmp_minimum_value = tmp_value;
  }

  return tmp_minimum_value;
}

real DatasetV1::get_max(size_t const data_start_index_received,
                      size_t const data_end_index_received,
                      size_t const input_index_received,
                      DATA::TYPE const data_type) const {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");
    return 0_r;
  } else if (data_start_index_received > data_end_index_received) {
    ERR(L"Start index (%zu) can not be greater than end index (%zu).",
        data_start_index_received, data_end_index_received);
    return 0_r;
  } else if (data_end_index_received > this->DatasetV1::get_n_data()) {
    ERR(L"End index (%zu) can not be greater than total examples (%zu).",
        data_end_index_received, this->DatasetV1::get_n_data());
    return 0_r;
  } else if (data_type > DATA::OUTPUT) {
    ERR(L"Type input (%d) is not managed in the function.", data_type);
    return 0_r;
  }

  size_t const tmp_input_size(data_type == DATA::INPUT ? this->p_n_inp
                                                       : this->p_n_out);
  size_t i, t;

  if (input_index_received >= tmp_input_size) {
    ERR(L"Input index (%zu) overflow (%zu).", input_index_received,
        tmp_input_size);
    return 0_r;
  }

  real const *const *const tmp_ptr_array_inputs(
      data_type == DATA::INPUT ? this->DatasetV1::Get__Input_Array()
                               : this->DatasetV1::Get__Output_Array());

  if (tmp_ptr_array_inputs == nullptr) {
    ERR(L"`tmp_ptr_array_inputs` is a nullptr.");
    return 0_r;
  }

  real tmp_maximum_value(-(std::numeric_limits<real>::max)()), tmp_input_value;

  for (i = data_start_index_received; i != data_end_index_received; ++i) {
    for (t = 0_UZ; t != this->p_seq_w; ++t) {
      tmp_input_value =
          tmp_ptr_array_inputs[i][t * tmp_input_size + input_index_received];

      if (tmp_input_value > tmp_maximum_value)
        tmp_maximum_value = tmp_input_value;
    }
  }

  return tmp_maximum_value;
}

real DatasetV1::get_max(size_t const data_start_index_received,
                      size_t const data_end_index_received,
                      DATA::TYPE const data_type) const {
  if (this->p_n_data == 0_UZ) {
    ERR(L"No data available.");
    return 0_r;
  } else if (data_start_index_received > data_end_index_received) {
    ERR(L"Start index (%zu) can not be greater than end index (%zu).",
        data_start_index_received, data_end_index_received);
    return 0_r;
  } else if (data_end_index_received > this->DatasetV1::get_n_data()) {
    ERR(L"End index (%zu) can not be greater than total examples (%zu).",
        data_end_index_received, this->DatasetV1::get_n_data());
    return 0_r;
  } else if (data_type > DATA::OUTPUT) {
    ERR(L"Type input (%d) is not managed in the function.", data_type);
    return 0_r;
  }

  size_t const tmp_input_size(data_type == DATA::INPUT ? this->p_n_inp
                                                       : this->p_n_out);
  size_t k;

  real tmp_value,
      tmp_maximum_value(this->get_max(
          data_start_index_received, data_end_index_received, 0_UZ, data_type));

  for (k = 1_UZ; k != tmp_input_size; ++k) {
    tmp_value = this->get_max(data_start_index_received,
                              data_end_index_received, k, data_type);

    if (tmp_value > tmp_maximum_value)
      tmp_maximum_value = tmp_value;
  }

  return (tmp_maximum_value);
}

real DatasetV1::get_inp(size_t const index_received,
                      size_t const sub_index_received) const {
  return this->Xm[index_received + this->p_str_i][sub_index_received];
}

real DatasetV1::get_out(size_t const index_received,
                      size_t const sub_index_received) const {
  return this->Ym[index_received + this->p_str_i][sub_index_received];
}

real const *const DatasetV1::get_inp(size_t const index_received) const {
  return this->Xm[index_received + this->p_str_i];
}

real const *const DatasetV1::get_out(size_t const index_received) const {
  return this->Ym[index_received + this->p_str_i];
}

real const *const *const DatasetV1::Get__Input_Array(void) const {
  return this->Xm + this->p_str_i;
}

real const *const *const DatasetV1::Get__Output_Array(void) const {
  return this->Ym + this->p_str_i;
}

size_t DatasetV1::Get__Sizeof(void) {
  size_t tmp_total_size_t(0);

  tmp_total_size_t += sizeof(DatasetV1);  // this

  if (this->_reference == false) {
    size_t k;

    if (this->Xm != nullptr)
      tmp_total_size_t += this->p_n_data * sizeof(real *);
    if (this->Ym != nullptr)
      tmp_total_size_t += this->p_n_data * sizeof(real *);

    if (this->X != nullptr)
      tmp_total_size_t +=
          this->p_n_data * this->p_n_inp * this->p_seq_w * sizeof(real);
    if (this->Y != nullptr)
      tmp_total_size_t +=
          this->p_n_data * this->p_n_out * this->p_seq_w * sizeof(real);

    if (this->_ptr_input_array_scaler__minimum_maximum != nullptr)
      tmp_total_size_t +=
          this->p_n_inp *
          sizeof(this->_ptr_input_array_scaler__minimum_maximum);
    if (this->_ptr_output_array_scaler__minimum_maximum != nullptr)
      tmp_total_size_t +=
          this->p_n_out *
          sizeof(this->_ptr_output_array_scaler__minimum_maximum);

    if (this->_ptr_input_array_scaler__zero_centered != nullptr)
      tmp_total_size_t +=
          this->p_n_inp * sizeof(this->_ptr_input_array_scaler__zero_centered);
    if (this->_ptr_output_array_scaler__zero_centered != nullptr)
      tmp_total_size_t +=
          this->p_n_out * sizeof(this->_ptr_output_array_scaler__zero_centered);

    if (this->Xm_coeff != nullptr) {
      tmp_total_size_t += this->p_n_inp * this->p_seq_w * sizeof(real *);

      if (this->Xm_coeff_size != nullptr)
        for (k = 0_UZ; k != this->p_n_inp; ++k)
          tmp_total_size_t +=
              this->Xm_coeff_size[k] * this->p_seq_w * sizeof(real);
    }

    if (this->Xm_coeff_size != nullptr)
      tmp_total_size_t += this->p_n_inp * sizeof(size_t);

    if (this->Ym_coeff != nullptr) {
      tmp_total_size_t += this->p_n_out * this->p_seq_w * sizeof(real *);

      if (this->Ym_coeff_size != nullptr)
        for (k = 0_UZ; k != this->p_n_out; ++k)
          tmp_total_size_t +=
              this->Ym_coeff_size[k] * this->p_seq_w * sizeof(real);
    }

    if (this->Ym_coeff_size != nullptr)
      tmp_total_size_t += this->p_n_out * sizeof(size_t);
  }

  return tmp_total_size_t;
}

ScalerMinMax *const DatasetV1::get_scaler_minmax(
    DATA::TYPE const data_type) const {
  switch (data_type) {
    case DATA::INPUT:
      return this->_ptr_input_array_scaler__minimum_maximum;
    case DATA::OUTPUT:
      return this->_ptr_output_array_scaler__minimum_maximum;
    default:
      ERR(L"Type input (%d) is not managed in the switch.", data_type);
      return nullptr;
  }
}

ScalerZeroCentered *const DatasetV1::Get__Scalar__Zero_Centered(
    DATA::TYPE const data_type) const {
  switch (data_type) {
    case DATA::INPUT:
      return this->_ptr_input_array_scaler__zero_centered;
    case DATA::OUTPUT:
      return this->_ptr_output_array_scaler__zero_centered;
    default:
      ERR(L"Type input (%d) is not managed in the switch.", data_type);
      return nullptr;
  }
}

DatasetV1::~DatasetV1(void) { this->Deallocate(); }
}  // namespace DL