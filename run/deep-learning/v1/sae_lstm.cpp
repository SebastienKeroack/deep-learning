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

// Deep learning:
#include "deep-learning/device/system/info.hpp"
#include "deep-learning/io/logger.hpp"
#include "deep-learning/io/term/input.hpp"
#include "deep-learning/v1/learner/models.hpp"

// Standard:
#ifdef _WIN32
#include <windows.h>
#elif __linux__
#include <cstring>
#endif
#include <thread>

using namespace DL;
using namespace DL::v1;
using namespace DL::Sys;
using namespace DL::Term;

bool preprocess_sae(class Datasets *const datasets) {
  size_t i;

  real min_inp((std::numeric_limits<real>::max)()),
      max_inp(-(std::numeric_limits<real>::max)());

  class DatasetV1 *const train_set(datasets->get_dataset(ENV::TRAIN));

  // Price.
  // Price, modwt.
  for (i = 0_UZ; i != 4_UZ; ++i) {
    if (datasets->preprocess_modwt(i, 3_UZ, DATA::INPUT) == false) {
      ERR(L"An error has been triggered from the "
          L"`preprocess_modwt()` function.");
      return false;
    }

    if (datasets->preprocess_modwt(i, 3_UZ, DATA::OUTPUT) == false) {
      ERR(L"An error has been triggered from the "
          L"`preprocess_modwt()` function.");
      return false;
    }
  }

  // Price, Get Min-Max.
  for (i = 0_UZ; i != 8_UZ; ++i) {
    min_inp = std::min<real>(
        train_set->get_min(0_UZ, train_set->DatasetV1::get_n_data(), i,
                           DATA::INPUT),
        min_inp);

    max_inp = std::max<real>(
        train_set->get_max(0_UZ, train_set->DatasetV1::get_n_data(), i,
                           DATA::INPUT),
        max_inp);
  }

  // Price, Set Min-Max.
  for (i = 0_UZ; i != 8_UZ; ++i) {
    if (datasets->preprocess_minmax(0_UZ, datasets->get_n_data(), i, min_inp,
                                    max_inp, 0_r, 1_r, DATA::INPUT) == false) {
      ERR(L"An error has been triggered from the "
          L"`preprocess_minmax()` function.");
      return false;
    }

    if (datasets->preprocess_minmax(0_UZ, datasets->get_n_data(), i, min_inp,
                                    max_inp, 0_r, 1_r, DATA::OUTPUT) == false) {
      ERR(L"An error has been triggered from the "
          L"`preprocess_minmax()` function.");
      return false;
    }
  }
  // |END| Price. |END|

  // RSI.
  if (datasets->preprocess_minmax(0_UZ, datasets->get_n_data(), 9_UZ, 0_r,
                                  100_r, 0_r, 1_r, DATA::INPUT) == false) {
    ERR(L"An error has been triggered from the "
        L"`preprocess_minmax()` function.");
    return false;
  }

  if (datasets->preprocess_minmax(0_UZ, datasets->get_n_data(), 9_UZ, 0_r,
                                  100_r, 0_r, 1_r, DATA::OUTPUT) == false) {
    ERR(L"An error has been triggered from the "
        L"`preprocess_minmax()` function.");
    return false;
  }
  // |END| RSI. |END|

  // ATR.
  min_inp = train_set->get_min(0_UZ, train_set->DatasetV1::get_n_data(), 10_UZ,
                               DATA::INPUT);
  max_inp = train_set->get_max(0_UZ, train_set->DatasetV1::get_n_data(), 10_UZ,
                               DATA::INPUT);

  if (datasets->preprocess_minmax(0_UZ, datasets->get_n_data(), 10_UZ, min_inp,
                                  max_inp, 0_r, 1_r, DATA::INPUT) == false) {
    ERR(L"An error has been triggered from the "
        L"`preprocess_minmax()` function.");
    return false;
  }

  if (datasets->preprocess_minmax(0_UZ, datasets->get_n_data(), 10_UZ, min_inp,
                                  max_inp, 0_r, 1_r, DATA::OUTPUT) == false) {
    ERR(L"An error has been triggered from the "
        L"`preprocess_minmax()` function.");
    return false;
  }
  // |END| ATR. |END|

  // StdDev.
  min_inp = train_set->get_min(0_UZ, train_set->DatasetV1::get_n_data(), 11_UZ,
                               DATA::INPUT);
  max_inp = train_set->get_max(0_UZ, train_set->DatasetV1::get_n_data(), 11_UZ,
                               DATA::INPUT);

  if (datasets->preprocess_minmax(0_UZ, datasets->get_n_data(), 11_UZ, min_inp,
                                  max_inp, 0_r, 1_r, DATA::INPUT) == false) {
    ERR(L"An error has been triggered from the "
        L"`preprocess_minmax()` function.");
    return false;
  }

  if (datasets->preprocess_minmax(0_UZ, datasets->get_n_data(), 11_UZ, min_inp,
                                  max_inp, 0_r, 1_r, DATA::OUTPUT) == false) {
    ERR(L"An error has been triggered from the "
        L"`preprocess_minmax()` function.");
    return false;
  }
  // |END| StdDev. |END|

  return true;
}

bool sae_lstm(void) {
  std::wstring const f_model_name(parse_wstring(L"Financial dataset name:"));
  std::wstring const s_model_name(parse_wstring(L"SAE model name:"));
  std::wstring const l_model_name(parse_wstring(L"LSTM model name:"));

#ifdef _WIN32
  if (SetConsoleTitleW(
          (f_model_name + L" - " + s_model_name + L" - " + l_model_name)
              .c_str()) == FALSE)
    WARN(L"Couldn't set a title to the console.");
#endif

  Models f_models, s_models, l_models;

  if (f_models.initialize_dirs(f_model_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`initialize_dirs(%ls, %ls)` function.",
        f_model_name.c_str(), f_model_name.c_str());
    return false;
  } else if (s_models.initialize_dirs(s_model_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`initialize_dirs(%ls, %ls)` function.",
        s_model_name.c_str(), s_model_name.c_str());
    return false;
  } else if (l_models.initialize_dirs(l_model_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`initialize_dirs(%ls, %ls)` function.",
        l_model_name.c_str(), l_model_name.c_str());
    return false;
  }

  // DatasetV1 Manager Parameters.
  DatasetsParams datasets_params;

  datasets_params.type_storage = 0;
  datasets_params.type_train = 0;

  if (f_models.initialize_datasets(&datasets_params) == false) {
    ERR(L"An error has been triggered from the "
        L"`initialize_datasets(ptr)` function.");
    return false;
  }

  datasets_params.type_storage = 2;
  datasets_params.type_train = 1;

  datasets_params.pct_train_size = 94.0;
  datasets_params.pct_valid_size = 4.0;

  datasets_params.train_params.value_0 = true;
  datasets_params.train_params.value_1 = 116;
  datasets_params.train_params.value_2 = 0;

  if (s_models.initialize_datasets(&datasets_params) == false) {
    ERR(L"An error has been triggered from the "
        L"`initialize_datasets(ptr)` function.");
    return false;
  } else if (l_models.initialize_datasets(&datasets_params) == false) {
    ERR(L"An error has been triggered from the "
        L"`initialize_datasets(ptr)` function.");
    return false;
  }
  // |END| DatasetV1 Manager Parameters. |END|

  // Memory all
  size_t const remaining_mem_(remaining_mem(10.0L, 1000_UZ * ::MEGABYTE));

  INFO(L"");
  INFO(L"Allowable memory:");
  INFO(L"Range[1, %zu] MBs.", remaining_mem_ / ::MEGABYTE);

  size_t const allowable_host_mem(
      parse_discrete(1_UZ, remaining_mem_ / ::MEGABYTE) * ::MEGABYTE);

  INFO(L"");
  // |END| Memory allocate. |END|

  // CUDA.
  size_t allowable_devc_mem(0_UZ);

#ifdef COMPILE_CUDA
  INFO(L"");
  if (accept(L"Do you want to use CUDA?")) {
    int device_id(-1);

    INFO(L"");
    s_models.set_use_cuda(
        CUDA__Input__Use__CUDA(device_id, allowable_devc_mem));
  }
#endif
  // |END| CUDA. |END|

  Model *s_model(nullptr), *l_model(nullptr);

  HIERARCHY::TYPE const hierarchy(HIERARCHY::TRAINER);

  bool const append_to_dset(accept(L"Append to the dataset?"));

  // load SAEs.
  if (s_models.load_model(HIERARCHY::TRAINER, allowable_host_mem,
                          allowable_devc_mem, false) == false) {
    ERR(L"An error has been triggered from the "
        L"`load_model(%ls, %zu, %zu, false)` function.",
        HIERARCHY_NAME[HIERARCHY::TRAINER].c_str(), allowable_host_mem,
        allowable_devc_mem);
    return false;
  } else if (s_models.load_model(HIERARCHY::TRAINED, allowable_host_mem,
                                 allowable_devc_mem, true) == false) {
    ERR(L"An error has been triggered from the "
        L"`load_model(%ls, %zu, %zu, true)` function.",
        HIERARCHY_NAME[HIERARCHY::TRAINED].c_str(), allowable_host_mem,
        allowable_devc_mem);
    return false;
  }

  s_model = s_models.get_model(HIERARCHY::TRAINED);
  // |END| load SAEs. |END|

  // Setup SAEs input/output mode.
  if (s_model->Set__Input_Mode(true) == false) {
    ERR(L"An error has been triggered from the "
        L"`Set__Input_Mode(true)` function.");
    return false;
  } else if (s_model->Set__Output_Mode(false) == false) {
    ERR(L"An error has been triggered from the "
        L"`Set__Output_Mode(false)` function.");
    return false;
  }

  // load LSTM.
  if (l_models.load_model(HIERARCHY::TRAINER, allowable_host_mem,
                          allowable_devc_mem, false) == false) {
    ERR(L"An error has been triggered from the "
        L"`load_model(%ls, %zu, %zu, false)` function.",
        HIERARCHY_NAME[HIERARCHY::TRAINER].c_str(), allowable_host_mem,
        allowable_devc_mem);
    return false;
  } else if (l_models.load_model(HIERARCHY::TRAINED, allowable_host_mem,
                                 allowable_devc_mem, true) == false) {
    ERR(L"An error has been triggered from the "
        L"`load_model(%ls, %zu, %zu, true)` function.",
        HIERARCHY_NAME[HIERARCHY::TRAINED].c_str(), allowable_host_mem,
        allowable_devc_mem);
    return false;
  }

  l_model = l_models.get_model(hierarchy);
  // |END| load LSTM. |END|

  Datasets const *const f_dsets(f_models.get_datasets());
  Datasets *const s_dsets(s_models.get_datasets()),
      *const l_dsets(l_models.get_datasets());

  // Validate input(s)/output(s) size.
  if (l_dsets->get_n_data() == 0_UZ) {
    ERR(L"No data available.");
    return false;
  } else if (l_dsets->get_n_data() != s_dsets->get_n_data()) {
    ERR(L"The number of data (%zu) differ from the number of "
        L"data received as argument (%zu).",
        l_dsets->get_n_data(), s_dsets->get_n_data());
    return false;
  } else if (l_dsets->get_seq_w() != s_dsets->get_seq_w()) {
    ERR(L"The number of recurrent depth (%zu) differ from the "
        L"number of recurrent depth received as argument (%zu).",
        l_dsets->get_seq_w(), s_dsets->get_seq_w());
    return false;
  } else if (s_dsets->valide_spec(s_model->n_inp, s_model->n_out,
                                  s_model->seq_w) == false) {
    ERR(L"An error has been triggered from the "
        L"`valide_spec(%zu, %zu, %zu)` function.",
        s_model->n_inp, s_model->n_out, s_model->seq_w);
    return false;
  } else if (s_model->type != MODEL::AUTOENCODER) {
    ERR(L"The neural network (%ls) receive as argument need to be "
        L"a %ls.",
        MODEL_NAME[s_model->type].c_str(),
        MODEL_NAME[MODEL::AUTOENCODER].c_str());
    return false;
  } else if (l_dsets->valide_spec(l_model->n_inp, l_model->n_out,
                                  l_model->seq_w) == false) {
    ERR(L"An error has been triggered from the "
        L"`valide_spec(%zu, %zu, %zu)` function.",
        l_model->n_inp, l_model->n_out, l_model->seq_w);
    return false;
  } else if (l_dsets->get_n_inp() != s_model->get_n_out()) {
    ERR(L"The number of input(s) (%zu) differ from the number of "
        L"output(s) from the autoencoder (%zu).",
        l_dsets->get_n_inp(), s_model->get_n_out());
    return false;
  }
  // |END| Validate input(s)/output(s) size. |END|

  // Input for optimization.
  INFO(L"");
  bool const train(accept(L"Do you want to optimize?"));

  size_t s_train_time(1_UZ), l_train_time(1_UZ);

  WhileCond while_cond;

  if (train) {
    INFO(L"SAEs optimization time.");
    INFO(L"default=120 (seconds).");
    s_train_time = parse_discrete(1_UZ, L"Time in seconds: ");

    INFO(L"");
    INFO(L"LSTM optimization time.");
    INFO(L"default=120 (seconds).");
    l_train_time = parse_discrete(1_UZ, L"Time in seconds: ");

    while_cond.type = WHILE_MODE::EXPIRATION;
  }
  // |END| Input for optimization. |END|

  // Preprocess SAEs dataset.
  if (preprocess_sae(s_dsets) == false) {
    ERR(L"An error has been triggered from the "
        L"`preprocess_sae(ptr)` function.");
    return false;
  }

  size_t const n_data(f_dsets->get_n_data()), seq_w(f_dsets->get_seq_w());
  size_t j, i, t, acc_avg_size, count_w(0_UZ), count_l(0_UZ);

  real const *f_inp;
  real *s_inp, *l_inp, *l_out, min_ohlc, max_ohlc;

  double sum_loss(0.0), loss(0.0), acc(0.0), acc_ema;

  std::thread s_thread, l_thread;

  // EMA.
  INFO(L"");
  INFO(L"Exponential moving average.");
  INFO(L"Range[1 , - ].");
  INFO(L"default=24.");
  acc_avg_size = parse_discrete(0_UZ, L"Time average: ");
  acc_ema = acc_avg_size == 0_UZ ? 0_r : 1_r / static_cast<real>(acc_avg_size);
  // |END| EMA. |END|

  s_inp = new real[s_model->n_inp * s_model->seq_w];
  l_inp = new real[l_model->n_inp * l_model->seq_w];
  l_out = new real[l_model->n_out * l_model->seq_w];

  INFO(L"");
  for (i = 0_UZ; i != n_data; ++i) {
    // Get inputs (financial data).
    f_inp = f_dsets->get_inp(i);
    memcpy(s_inp, f_inp, seq_w * f_dsets->get_n_inp() * sizeof(real));

    // Preprocess SAEs.
    {
      min_ohlc = s_dsets->get_scaler_minmax(DATA::INPUT)[0].minval;
      max_ohlc = s_dsets->get_scaler_minmax(DATA::INPUT)[0].maxval;

      // Price.
      //  Price, Min-Max.
      for (j = 0_UZ; j != 8_UZ; ++j) {
        if (s_dsets->preprocess_minmax(j, s_inp, DATA::INPUT) == false) {
          ERR(L"An error has been triggered from the "
              L"`preprocess_minmax()` function.");
          return false;
        }
      }

      //  Price, modwt.
      for (j = 0_UZ; j != 4_UZ; ++j) {
        // Inverse preprocess min-max before preprocessing modwt.
        if (s_dsets->preprocess_minmax_inv(j, DATA::INPUT) == false) {
          ERR(L"An error has been triggered from the "
              "`preprocess_minmax_inv()` function.");
          return false;
        }

        // Inverse preprocess modwt before preprocessing the inputs receive as
        // argument.
        if (s_dsets->preprocess_modwt_inv(j, DATA::INPUT) == false) {
          ERR(L"An error has been triggered from the "
              "`preprocess_modwt_inv()` function.");
          return false;
        }

        // Preprocess array receive as arguments to remove noise.
        if (s_dsets->preprocess_modwt(3_UZ, s_inp, DATA::INPUT) == false) {
          ERR(L"An error has been triggered from the "
              "`preprocess_modwt()` function.");
          return false;
        }

        // Re-preprocess dataset to remove noise.
        if (s_dsets->preprocess_modwt(j, 3_UZ, DATA::INPUT) == false) {
          ERR(L"An error has been triggered from the "
              "`preprocess_modwt()` function.");
          return false;
        }

        // Re-preprocess min-max with past parameters after preprocessing modwt.
        if (s_dsets->preprocess_minmax(0_UZ, s_dsets->get_n_data(), j, min_ohlc,
                                       max_ohlc, 0_r, 1_r,
                                       DATA::INPUT) == false) {
          ERR(L"An error has been triggered from the "
              "`preprocess_minmax()` function.");
          return false;
        }
      }
      // |END| Price. |END|

      // RSI.
      if (s_dsets->preprocess_minmax(9_UZ, s_inp, DATA::INPUT) == false) {
        ERR(L"An error has been triggered from the "
            "`preprocess_minmax()` function.");
        return false;
      }
      // |END| RSI. |END|

      // ATR.
      if (s_dsets->preprocess_minmax(10_UZ, s_inp, DATA::INPUT) == false) {
        ERR(L"An error has been triggered from the "
            "`preprocess_minmax()` function.");
        return false;
      }
      // |END| ATR. |END|

      // StdDev.
      if (s_dsets->preprocess_minmax(11_UZ, s_inp, DATA::INPUT) == false) {
        ERR(L"An error has been triggered from the "
            "`preprocess_minmax()` function.");
        return false;
      }
      // |END| StdDev. |END|
    }
    // |END| Preprocess SAEs. |END|

    // Propagate inputs into the SAEs.
    s_model->forward_pass(1_UZ, &s_inp);

    // Get inputs (SAEs outputs).
    for (t = 0_UZ; t != seq_w; ++t)
      memcpy(l_inp + t * l_model->n_inp, s_model->get_out(0_UZ, t),
             l_model->n_inp * sizeof(real));

    // Propagate inputs into the LSTM.
    l_model->forward_pass(1_UZ, &l_inp);

    // Compute LSTM loss.
    {
      // Get outputs (financial data).
      memcpy(l_out, f_dsets->get_out(i),
             seq_w * f_dsets->get_n_out() * sizeof(real));

      // reset loss before proceeding to the computation of the loss.
      l_model->reset_loss();

      // Compute the loss with l1 function.
      l_model->set_loss_fn(LOSS_FN::L2);

      // Compute LSTM loss.
      l_model->compute_loss(1_UZ, &l_out);

      // Set accuracy trial.
      l_model->n_acc_trial = f_dsets->get_n_out();

      // State of the trade.
      if (l_model->get_accu(ENV::NONE) == 100_r)
        ++count_w;
      else
        ++count_l;

      // Setup loss variable.
      l_model->set_loss_fn(LOSS_FN::RMSE);
      *l_model->ptr_array_number_loss += i * f_dsets->get_n_out();
      sum_loss = *l_model->ptr_array_loss_values += sum_loss;

      // Get loss.
      loss = l_model->get_loss(ENV::NONE);

      // Exponential moving average.
      if (i >= acc_avg_size && acc_ema != 0_r) {
        acc += acc_ema * (l_model->get_accu(ENV::NONE) - acc);

        INFO(L"T[%zu]: [L:%f | A:%f] --- [W:%zu vs L:%zu], Output[0]: %f.", i,
             loss, acc, count_w, count_l,
             cast(l_model->get_out(0_UZ, seq_w - 1_UZ)[0]));
      }
      // Moving average.
      else {
        acc += l_model->get_accu(ENV::NONE);

        INFO(L"T[%zu]: [L:%f | A:%f] --- [W:%zu vs L:%zu], Output[0]: %f.", i,
             loss, acc / static_cast<real>(i + 1_UZ), count_w, count_l,
             cast(l_model->get_out(0_UZ, seq_w - 1_UZ)[0]));
      }

      // Print output(s)
      for (size_t k(0_UZ); k != f_dsets->get_n_out(); ++k) {
        INFO(L"Output[%zu]: %f", k,
             cast(l_model->get_out(0_UZ, seq_w - 1_UZ)[k]));
      }
    }
    // |END| Compute LSTM loss. |END|

    // Inverse preprocess SAEs.
    {
      // Price.
      // Price, Min-Max.
      for (j = 0_UZ; j != 8_UZ; ++j) {
        if (s_dsets->preprocess_minmax_inv(j, DATA::INPUT) == false) {
          ERR(L"An error has been triggered from the "
              "`preprocess_minmax_inv()` function.");
          return false;
        }

        if (s_dsets->preprocess_minmax_inv(j, DATA::OUTPUT) == false) {
          ERR(L"An error has been triggered from the "
              "`preprocess_minmax_inv()` function.");
          return false;
        }
      }

      // Price, modwt.
      for (j = 0_UZ; j != 4_UZ; ++j) {
        if (s_dsets->preprocess_modwt_inv(j, DATA::INPUT) == false) {
          ERR(L"An error has been triggered from the "
              "`preprocess_modwt_inv()` function.");
          return false;
        }

        if (s_dsets->preprocess_modwt_inv(j, DATA::OUTPUT) == false) {
          ERR(L"An error has been triggered from the "
              "`preprocess_modwt_inv()` function.");
          return false;
        }
      }
      // |END| Price. |END|

      // RSI.
      if (s_dsets->preprocess_minmax_inv(9_UZ, DATA::INPUT) == false) {
        ERR(L"An error has been triggered from the "
            "`preprocess_minmax_inv()` function.");
        return false;
      }

      if (s_dsets->preprocess_minmax_inv(9_UZ, DATA::OUTPUT) == false) {
        ERR(L"An error has been triggered from the "
            "`preprocess_minmax_inv()` function.");
        return false;
      }
      // |END| RSI. |END|

      // ATR.
      if (s_dsets->preprocess_minmax_inv(10_UZ, DATA::INPUT) == false) {
        ERR(L"An error has been triggered from the "
            "`preprocess_minmax_inv()` function.");
        return false;
      }

      if (s_dsets->preprocess_minmax_inv(10_UZ, DATA::OUTPUT) == false) {
        ERR(L"An error has been triggered from the "
            "`preprocess_minmax_inv()` function.");
        return false;
      }
      // |END| ATR. |END|

      // StdDev.
      if (s_dsets->preprocess_minmax_inv(11_UZ, DATA::INPUT) == false) {
        ERR(L"An error has been triggered from the "
            "`preprocess_minmax_inv()` function.");
        return false;
      }

      if (s_dsets->preprocess_minmax_inv(11_UZ, DATA::OUTPUT) == false) {
        ERR(L"An error has been triggered from the "
            "`preprocess_minmax_inv()` function.");
        return false;
      }
      // |END| StdDev. |END|
    }
    // |END| Inverse preprocess SAEs. |END|

    // Append into the SAE dataset.
    if (append_to_dset && s_models.append_to_dataset(f_inp, f_inp) == false) {
      ERR(L"An error has been triggered from the "
          "`append_to_dataset()` function.");
      return false;
    }

    // Preprocess SAEs dataset.
    if (preprocess_sae(s_dsets) == false) {
      ERR(L"An error has been triggered from the "
          "`preprocess_sae(ptr)` function.");
      return false;
    }

    // optimize.
    if (train) {
      // Append into the LSTM dataset.
      if (append_to_dset && l_models.append_to_dataset(l_inp, l_out) == false) {
        ERR(L"An error has been triggered from the "
            "`append_to_dataset()` function.");
        return false;
      }

      // Set expiration SAEs optimization.
      while_cond.expiration =
          std::chrono::system_clock::now() + std::chrono::seconds(s_train_time);
      if (s_models.set_while_cond(while_cond) == false) {
        ERR(L"An error has been triggered from the "
            "`set_while_cond()` function.");
        return false;
      }

      // Evaluate all envs using the SAEs model.
      if (s_models.if_require_evaluate_envs_pre_train() == false) {
        ERR(L"An error has been triggered from the "
            "`if_require_evaluate_envs_pre_train()` function.");
        return false;
      }

      // Optimize SAEs.
      s_thread = std::thread([&models = s_models]() { models.pre_training(); });

      // Set expiration LSTM optimization.
      while_cond.expiration =
          std::chrono::system_clock::now() + std::chrono::seconds(l_train_time);
      if (l_models.set_while_cond(while_cond) == false) {
        ERR(L"An error has been triggered from the "
            "`set_while_cond()` function.");
        return false;
      }

      // Evaluate all envs using the LSTM model.
      if (l_models.if_require_evaluate_envs() == false) {
        ERR(L"An error has been triggered from the "
            "`if_require_evaluate_envs()` function.");
        return false;
      }

      // Optimize LSTM.
      l_thread = std::thread(&Models::optimize, &l_models);

      auto join_n_compare_fn([](std::thread &thread, Models &models) -> void {
        if (thread.joinable()) {
          thread.join();
          models.compare_trained();
        }
      });

      // join the SAEs.
      if (s_thread.joinable()) {
        s_thread.join();

        // SAE compare trained.
        if (s_models.compare_trained_pre_train()) {
          // join and compare the LSTM.
          join_n_compare_fn(l_thread, l_models);

          // Update the LSTM dataset.
          if (l_dsets->replace_entries(s_dsets, s_model) == false) {
            ERR(L"An error has been triggered from the "
                "`replace_entries(ptr, ptr)` function.");
            return false;
          }
        }
      }

      // join and compare the LSTM.
      join_n_compare_fn(l_thread, l_models);
    }

    // Convert moving average to exponential moving average.
    if (i + 1_UZ == acc_avg_size && acc_ema != 0_r) {
      acc /= static_cast<real>(i + 1_UZ);
    }
  }

  delete[] (l_out);
  delete[] (l_inp);
  delete[] (s_inp);

  return true;
}
