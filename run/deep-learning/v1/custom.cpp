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

// PCH:
#include "run/pch.hpp"

// File header:
#include "run/deep-learning/v1/custom.hpp"

// Deep learning:
#include "deep-learning/data/string.hpp"
#include "deep-learning/io/file.hpp"
#include "deep-learning/io/logger.hpp"
#include "deep-learning/io/term/input.hpp"
#include "deep-learning/v1/learner/models.hpp"
#include "deep-learning/v1/mem/reallocate.hpp"

// Standard:
#ifdef _WIN32
#include <windows.h>
#endif

using namespace DL;
using namespace DL::v1;
using namespace DL::Str;

class Experimental {
 protected:
#ifdef COMPILE_CUDA
  struct cudaDeviceProp cudevice;
#endif

  bool const use_cuda = false;
  bool const use_mp = false;
  bool const online_training = false;
  bool const load_spec_params = false;
  bool const use_dropout = false;
  bool const use_normalization = true;
  bool const use_bn_before_af = true;
  bool const use_warm_restarts = false;
  bool const use_bidirectional = false;
  bool const use_residual = true;
  bool const pooling_ceil_mode = false;
  bool const clip_gradient = false;
  bool const tied_params = false;
  bool const save_params = false;
  bool const copy_params = true;
  bool const use_train_set = true;
  bool const use_valid_set = false;
  bool const use_infer_set = true;
  bool const use_adept = false;
  bool const infer_set_print_x = false;
  bool const infer_set_print_y = false;
  bool const print_params = false;
  bool use_pooling = true;
  bool use_bottleneck = false;
  bool update_bn_final = true;

  int const device_id = 0;

  unsigned int const dist_seed = 5413u;

  long long int wf_alphas[2] = {-9ll, 0ll};

  size_t const n_layers = 12_UZ;
  size_t const N_units[2] = {32_UZ, 1_UZ};
  size_t const residual_block_width = 3_UZ;
  size_t const pooling_kernel_size = 2_UZ;
  size_t const pooling_stride = 2_UZ;
  size_t const pooling_padding = 0_UZ;
  size_t const pooling_dilation = 0_UZ;
  size_t const seq_w = 27_UZ;
  size_t const n_episodes = 10_UZ;
  size_t const n_epochs = 10_UZ;
  size_t const input_image_size = 28_UZ;
  size_t const allowable_mem = 256_UZ * ::MEGABYTE;
  size_t n_models = 1_UZ;

  real const dropout_values[2] = {0.5_r, 0.5_r};
  real const l1 = 0_r;
  real const l2 = 0_r;
  real const max_norm = 8_r;
  real const weight_decay = 0_r;
  real const acc_var = 0.49_r;
  real const lr = 1e-3_r;
  real const lr_final = 1e-1_r;
  real const weight_minval_range = -1_r;
  real const weight_maxval_range = 1_r;
  real const clip_gradient_val = 1_r;
  real const wm_min_lr = 1e-9_r;
  real const wm_max_lr = 0.01_r;
  real const wm_initial_ti = 1_r;
  real const wm_multiplier = 2_r;
  real const bn_mom_avg = 1_r;
  real const bn_epsilon = 1e-1_r;

  double pct_threads = 25.0;

  MODEL::TYPE const type_model = MODEL::FEEDFORWARD;
  LAYER::TYPE const type_fully_connected =
      LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT;
  LAYER::TYPE const type_pooling_layer = LAYER::MAX_POOLING;
  ACTIVATION::TYPE const type_af_hidden = ACTIVATION::ISRLU;
  ACTIVATION::TYPE const type_af_output = ACTIVATION::SOFTMAX;
  LOSS_FN::TYPE const type_loss_fn = LOSS_FN::CROSS_ENTROPY;
  ACCU_FN::TYPE const type_acc_fn = ACCU_FN::CROSS_ENTROPY;
  OPTIMIZER::TYPE const type_optimizer = OPTIMIZER::AMSGRAD;
  INITIALIZER::TYPE const type_weight_initializer = INITIALIZER::ORTHOGONAL;
  LAYER_NORM::TYPE const type_normalization = LAYER_NORM::BATCH_NORMALIZATION;
  LAYER_DROPOUT::TYPE const type_dropout = LAYER_DROPOUT::SHAKEDROP;

  std::wstring dset_path_name = L"sMNIST_28";

 public:
  Experimental(bool const nvvp_or_nsight);
  ~Experimental(void);

#ifdef COMPILE_CUDA
  void setup_cuda(void);
  void print_cudevice(void);
  bool setup_cudevice(void);
#endif  // COMPILE_CUDA

  void print_attributes(void);
  void start(void);

  bool run(unsigned int model_id);
  bool check_attributes(void);
  bool setup_datasets(void);

 private:
  size_t _mem_allocate = 0_UZ;

  var *_initial_parameters = nullptr;

  double _past_loss = 0.0;

  class Datasets *_datasets;
};

Experimental::Experimental(bool const nvvp_or_nsight) {
  if (nvvp_or_nsight) this->n_models = 1_UZ;

  this->use_pooling &= this->use_residual;
  this->use_bottleneck &= this->use_residual;
  this->update_bn_final &= this->use_normalization;
}

Experimental::~Experimental(void) {
  delete (this->_datasets);
  delete[] (this->_initial_parameters);
}

#ifdef COMPILE_CUDA
void Experimental::setup_cuda(void) {
  if (this->use_cuda) return;

  CUDA__Set__Device(this->device_id);
  CUDA__Reset();
  CUDA__Set__Synchronization_Depth(3_UZ);
  CUDA__Safe_Call(
      cudaDeviceSetLimit(cudaLimit::cudaLimitDevRuntimeSyncDepth, 3u));
  CUDA__Safe_Call(cudaGetDeviceProperties(&this->cudevice,
                                          static_cast<int>(this->device_id)));
}

void Experimental::print_cudevice(void) {
  if (this->use_cuda) return;

  CUDA__Print__Device_Property(this->cudevice, this->device_id);
}

bool Experimental::setup_cudevice(void) {
  if (this->use_cuda) return true;

  size_t mem_total(0), mem_free(0);

  CUDA__Safe_Call(cudaMemGetInfo(&mem_free, &mem_total));

  this->_mem_allocate =
      mem_free < this->allowable_mem ? mem_free : this->allowable_mem;

  std::ostringstream o("");
  o << L"--- GPU | Memory ---" << CRLF;
  o << L"ID: " << this->device_id << CRLF;
  o << L"Available: " << mem_free / MEGABYTE << L"MB(s)." << CRLF;
  o << L"Used: " << (mem_total - mem_free) / MEGABYTE << L"MB(s)." << CRLF;
  o << L"allocate: " << this->_mem_allocate / MEGABYTE << L"MB(s)." << CRLF;
  o << L"Total: " << mem_total / MEGABYTE << L"MB(s)." << CRLF;
  INFO(o.str().c_str());

  CUDA__Initialize__Device(this->cudevice, this->_mem_allocate);
  // |END| Memory allocate. |END|

  if (this->_datasets->Initialize__CUDA() == false) {
    ERR(L"An error has been triggered from the "
        L"`Initialize__CUDA` function.");
    return false;
  }

  return true
}
#endif

void Experimental::print_attributes(void) {
  std::wostringstream o(L"");

  o << L"--- CUDA ---" << CRLF;
  o << L"Use: " << to_wstring(this->use_cuda) << CRLF;
  o << L"GPU ID: " << this->device_id << CRLF;

  o << L"--- OpenMP ---" << CRLF;
  o << L"Use: " << to_wstring(this->use_mp) << CRLF;
  o << L"Maximum cpu usage: " << this->pct_threads << '%' << CRLF;

  o << L"--- Program ---" << CRLF;
  o << L"N run: " << this->n_models << CRLF;
  o << L"N episodes per run: " << this->n_episodes << CRLF;
  o << L"N epochs per episode: " << this->n_epochs << CRLF;
  o << L"Adept: " << to_wstring(this->use_adept) << CRLF;

  o << L"--- DatasetV1 ---" << CRLF;
  o << L"Name: " << this->dset_path_name << CRLF;
  o << L"Train set: " << to_wstring(this->use_train_set) << CRLF;
  o << L"Valid set: " << to_wstring(this->use_valid_set) << CRLF;
  o << L"Testg set: " << to_wstring(this->use_infer_set) << CRLF;
  o << L"Print X from Testg set: " << to_wstring(this->infer_set_print_x)
    << CRLF;
  o << L"Print Y from Testg set: " << to_wstring(this->infer_set_print_y)
    << CRLF;
  o << L"Image size: " << this->input_image_size << CRLF;
  o << L"Simulate online training: " << to_wstring(this->online_training)
    << CRLF;
  o << L"Sequence window: " << this->seq_w << CRLF;

  o << L"--- Model ---" << CRLF;
  o << L"Type: " << MODEL_NAME[this->type_model] << CRLF;
  o << L"N layer(s): " << this->n_layers << CRLF;
  o << L"N unit(s)[0]: " << this->N_units[0] << CRLF;
  o << L"N unit(s)[1]: " << this->N_units[1] << CRLF;
  o << L"Bidirectional: " << to_wstring(this->use_bidirectional) << CRLF;
  o << L"Hidden layer AF: " << ACTIVATION_NAME[this->type_af_hidden] << CRLF;
  o << L"Max mem allocate: " << this->allowable_mem << L" byte(s)." << CRLF;

  o << L"--- Parameters ---" << CRLF;
  o << L"Weight min range: " << this->weight_minval_range << CRLF;
  o << L"Weight max range: " << this->weight_maxval_range << CRLF;
  o << L"Init: " << INITIALIZER_NAME[this->type_weight_initializer] << CRLF;
  o << L"Tied: " << to_wstring(this->tied_params) << CRLF;
  o << L"load: " << to_wstring(this->load_spec_params) << CRLF;
  o << L"save: " << to_wstring(this->save_params) << CRLF;
  o << L"copy: " << to_wstring(this->copy_params) << CRLF;
  o << L"Print: " << to_wstring(this->print_params) << CRLF;

  o << L"--- Residual ---" << CRLF;
  o << L"Use: " << to_wstring(this->use_residual) << CRLF;
  o << L"Pooling: " << to_wstring(this->use_pooling) << CRLF;
  o << L"Bottleneck: " << to_wstring(this->use_bottleneck) << CRLF;
  o << L"Block width: " << this->residual_block_width << CRLF;
  o << L"Widening factor, alpha[0]: " << this->wf_alphas[0] << CRLF;
  o << L"Widening factor, alpha[1]: " << this->wf_alphas[1] << CRLF;

  o << L"--- Dense layer ---" << CRLF;
  o << L"Type: " << LAYER_NAME[this->type_fully_connected] << CRLF;
  o << L"Output layer AF: " << ACTIVATION_NAME[this->type_af_output] << CRLF;

  o << L"--- Pooling layer ---" << CRLF;
  o << L"Type: " << LAYER_NAME[this->type_pooling_layer] << CRLF;
  o << L"Ceil mode: " << to_wstring(this->pooling_ceil_mode) << CRLF;
  o << L"Kernel size: " << this->pooling_kernel_size << CRLF;
  o << L"Stride: " << this->pooling_stride << CRLF;
  o << L"Padding: " << this->pooling_padding << CRLF;
  o << L"Dilation: " << this->pooling_dilation << CRLF;

  o << L"--- Normalization ---" << CRLF;
  o << L"Use: " << to_wstring(this->use_normalization) << CRLF;
  o << L"Type: " << LAYER_NORM_NAME[this->type_normalization] << CRLF;
  o << L"Use before AF: " << to_wstring(this->use_bn_before_af) << CRLF;
  o << L"Update BN pop final: " << to_wstring(this->update_bn_final) << CRLF;
  o << L"Momentum average: " << this->bn_mom_avg << CRLF;
  o << L"Epsilon: " << this->bn_epsilon << CRLF;

  o << L"--- Dropout ---" << CRLF;
  o << L"Use: " << to_wstring(this->use_dropout) << CRLF;
  o << L"Type: " << LAYER_DROPOUT_NAME[this->type_dropout] << CRLF;
  o << L"Rate[0]: " << this->dropout_values[0] << CRLF;
  o << L"Rate[1]: " << this->dropout_values[1] << CRLF;

  o << L"--- Optimizer ---" << CRLF;
  o << L"Name: " << OPTIMIZER_NAME[this->type_optimizer] << CRLF;
  o << L"lr: " << this->lr << CRLF;
  o << L"lr, final: " << this->lr_final << CRLF;
  o << L"L1: " << this->l1 << CRLF;
  o << L"L2: " << this->l2 << CRLF;
  o << L"Max-norm: " << this->max_norm << CRLF;
  o << L"Weight decay: " << this->weight_decay << CRLF;

  o << L"--- Warm restarts ---" << CRLF;
  o << L"Use: " << to_wstring(this->use_warm_restarts) << CRLF;
  o << L"Minimum learning rate: " << this->wm_min_lr << CRLF;
  o << L"Maximum learning rate: " << this->wm_max_lr << CRLF;
  o << L"Initial Ti: " << this->wm_initial_ti << CRLF;
  o << L"Multiplier: " << this->wm_multiplier << CRLF;

  o << L"--- Clip gradients ---" << CRLF;
  o << L"Use: " << to_wstring(this->clip_gradient) << CRLF;
  o << L"Val: " << this->clip_gradient_val << CRLF;

  o << L"--- Loss function ---" << CRLF;
  o << L"Type: " << LOSS_FN_NAME[this->type_loss_fn] << CRLF;

  o << L"--- Accuracy function ---" << CRLF;
  o << L"Type: " << ACC_FN_NAME[this->type_acc_fn] << CRLF;
  o << L"Variance: " << this->acc_var << CRLF;

  INFO(o.str().c_str());
}

void Experimental::start(void) {
  for (unsigned int i = 0u; i != this->n_models; ++i) {
    if (this->run(i) == false) {
      ERR(L"An error has been triggered from the "
          L"`Experimental::run(%u)` function.", i);
      break;
    }
  }
}

bool Experimental::run(unsigned int model_id) {
  std::wstring const path_params(this->dset_path_name + L".net"),
      path_spec_params(this->dset_path_name + L".nn");

  size_t ep, tmp_sub_index;

  double tmp_compute_time, tmp_time_total(0.0),
      tmp_widening_factors[2] = {0}, tmp_widening_factor_units[2] = {0};

  Model *model(new Model);

  Layer_Parameters layer_params, layer_params_widening;

  std::vector<Layer_Parameters> layers_params;

  TIME_POINT time_str, time_end;

  layers_params.clear();

  INFO(L"");
  INFO(L"Run #%zu", model_id);

  if (this->load_spec_params) {
    INFO(L"Neural network: Loading.");

    if (File::path_exist(path_params) == false) {
      ERR(L"Could not find the following path \"%ls\".", path_params.c_str());
      return false;
    }

    if (File::path_exist(path_spec_params) == false) {
      ERR(L"Could not find the following path \"%ls\".",
          path_spec_params.c_str());
      return false;
    }

    INFO(L"Neural network: load from %ls.", path_params.c_str());
    INFO(L"Neural network: load from %ls.", path_spec_params.c_str());

    if (model->load(path_params, path_spec_params, this->allowable_mem) ==
        false) {
      ERR(L"An error has been triggered from the `load(%ls, "
          L"%ls, %zu)` function.",
          path_params.c_str(), path_spec_params.c_str(), this->allowable_mem);

      INFO(L"DatasetV1: Deallocate.");
      delete (this->_datasets);
      delete[] (this->_initial_parameters);

      return false;
    }
  } else {
    layer_params.type_layer = LAYER::FULLY_CONNECTED;
    layer_params.unit_parameters[0] = this->_datasets->get_n_inp();
    layers_params.push_back(layer_params);

    INFO(L"Neural network: compile.");
    switch (this->type_model) {
      case MODEL::AUTOENCODER:
        for (ep = 1_UZ; ep != this->n_layers - 1_UZ; ++ep) {
          layer_params.type_layer = LAYER::FULLY_CONNECTED;
          layer_params.unit_parameters[0] = this->N_units[0];
          layers_params.push_back(layer_params);

          INFO(L"Layer[%zu]: Type: %ls.", ep,
               LAYER_NAME[layer_params.type_layer].c_str());

          INFO(L"Layer[%zu]: Number neuron unit(s): %zu.", ep,
               layer_params.unit_parameters[0]);
        }
        break;
      case MODEL::FEEDFORWARD:
        if (this->use_residual) {
          if (this->n_layers < 6_UZ) {
            ERR(L"The number of layer(s) (%zu) can not be less "
                L"than 6.",
                this->n_layers);

            delete (model);

            return false;
          }

          size_t const n_residual_layers((this->n_layers - 3_UZ) / 3_UZ),
              n_residual_layers_last_group((this->n_layers - 3_UZ) -
                                           2_UZ * n_residual_layers);
          size_t residual_unit_i;

          struct Neural_Network_Initializer tmp_Neural_Network_Initializer;

          tmp_widening_factors[0] =
              static_cast<double>(this->wf_alphas[0]) /
              static_cast<double>(2_UZ * n_residual_layers +
                                  n_residual_layers_last_group);

          // First hidden layer.
          layer_params.type_layer = this->type_fully_connected;
          layer_params.unit_parameters[0] = this->N_units[0];
          layers_params.push_back(layer_params);

          INFO(L"Layer[1]: Type: %ls.",
               LAYER_NAME[layer_params.type_layer].c_str());
          INFO(L"Layer[1]: Number neuron unit(s): %zu.",
               layer_params.unit_parameters[0]);
          // |END| First hidden layer. |END|

          // Residual group #1.
          tmp_widening_factor_units[0] = static_cast<double>(this->N_units[0]);

          for (residual_unit_i = 0_UZ; residual_unit_i != n_residual_layers;
               ++residual_unit_i) {
            // Residual unit.
            layer_params.type_layer = LAYER::RESIDUAL;
            layer_params.unit_parameters[0] = this->residual_block_width;

            layers_params.push_back(layer_params);

            INFO(L"Layer[%zu]: Type: %ls.", layers_params.size() - 1_UZ,
                 LAYER_NAME[layer_params.type_layer].c_str());
            INFO(L"Layer[%zu]: Block depth: %zu.", layers_params.size() - 1_UZ,
                 layer_params.unit_parameters[0]);
            // |END| Residual unit. |END|

            // Building block.
            tmp_sub_index = 0_UZ;
            layer_params.type_layer = this->type_fully_connected;

            if (this->use_bottleneck) {
              // First hidden layer inside the residual block.
              layer_params.unit_parameters[0] =
                  static_cast<size_t>(tmp_widening_factor_units[0]);

              layers_params.push_back(layer_params);

              INFO(L"Layer[%zu]: Type: %ls.", layers_params.size() - 1_UZ,
                   LAYER_NAME[layer_params.type_layer].c_str());
              INFO(L"Layer[%zu]: Number neuron unit(s): %zu.",
                   layers_params.size() - 1_UZ,
                   layer_params.unit_parameters[0]);

              ++tmp_sub_index;
              // |END| First hidden layer inside the residual block. |END|

              // Second hidden layer inside the residual block.
              layer_params.unit_parameters[0] = static_cast<size_t>(
                  std::max<double>(
                      tmp_widening_factor_units[0],
                      tmp_widening_factor_units[0] + tmp_widening_factors[0]) /
                  2.0);

              layers_params.push_back(layer_params);

              INFO(L"Layer[%zu]: Type: %ls.", layers_params.size() - 1_UZ,
                   LAYER_NAME[layer_params.type_layer].c_str());
              INFO(L"Layer[%zu]: Number neuron unit(s): %zu.",
                   layers_params.size() - 1_UZ,
                   layer_params.unit_parameters[0]);

              ++tmp_sub_index;
              // |END| Second hidden layer inside the residual block. |END|
            }

            tmp_widening_factor_units[0] += tmp_widening_factors[0];
            layer_params.unit_parameters[0] =
                static_cast<size_t>(tmp_widening_factor_units[0]);

            for (; tmp_sub_index != this->residual_block_width;
                 ++tmp_sub_index) {
              layers_params.push_back(layer_params);

              INFO(L"Layer[%zu]: Type: %ls.", layers_params.size() - 1_UZ,
                   LAYER_NAME[layer_params.type_layer].c_str());
              INFO(L"Layer[%zu]: Number neuron unit(s): %zu.",
                   layers_params.size() - 1_UZ,
                   layer_params.unit_parameters[0]);
            }
            // |END| Building block. |END|
          }
          // |END| Residual group #1. |END|

          // Residual group #2.
          //  Pooling layer.
          if (this->use_pooling) {
            layer_params.type_layer = this->type_pooling_layer;
            layer_params.unit_parameters[0] = this->pooling_kernel_size;
            layer_params.unit_parameters[1] = this->pooling_stride;
            layer_params.unit_parameters[2] = this->pooling_padding;
            layer_params.unit_parameters[3] = this->pooling_dilation;
            layer_params.unit_parameters[4] =
                static_cast<size_t>(this->pooling_ceil_mode);
            layers_params.push_back(layer_params);

            INFO(L"Layer[%zu]: Type: %ls.", layers_params.size() - 1_UZ,
                 LAYER_NAME[layer_params.type_layer].c_str());
            INFO(L"Layer[%zu]: Kernel size: %zu.", layers_params.size() - 1_UZ,
                 layer_params.unit_parameters[0]);
            INFO(L"Layer[%zu]: Stride: %zu.", layers_params.size() - 1_UZ,
                 layer_params.unit_parameters[1]);
            INFO(L"Layer[%zu]: Padding: %zu.", layers_params.size() - 1_UZ,
                 layer_params.unit_parameters[2]);
            INFO(L"Layer[%zu]: Dilation: %zu.", layers_params.size() - 1_UZ,
                 layer_params.unit_parameters[3]);
            INFO(L"Layer[%zu]: Ceil mode: %ls.", layers_params.size() - 1_UZ,
                 layer_params.unit_parameters[4] != 0_UZ ? "true" : "false");
          }
          //  |END| Pooling layer. |END|

          for (residual_unit_i = 0_UZ; residual_unit_i != n_residual_layers;
               ++residual_unit_i) {
            // Residual unit.
            layer_params.type_layer = LAYER::RESIDUAL;
            layer_params.unit_parameters[0] = this->residual_block_width;

            layers_params.push_back(layer_params);

            INFO(L"Layer[%zu]: Type: %ls.", layers_params.size() - 1_UZ,
                 LAYER_NAME[layer_params.type_layer].c_str());
            INFO(L"Layer[%zu]: Block depth: %zu.", layers_params.size() - 1_UZ,
                 layer_params.unit_parameters[0]);
            // |END| Residual unit. |END|

            // Building block.
            tmp_sub_index = 0_UZ;
            layer_params.type_layer = this->type_fully_connected;

            if (this->use_bottleneck) {
              // First hidden layer inside the residual block.
              layer_params.unit_parameters[0] =
                  static_cast<size_t>(tmp_widening_factor_units[0]);

              layers_params.push_back(layer_params);

              INFO(L"Layer[%zu]: Type: %ls.", layers_params.size() - 1_UZ,
                   LAYER_NAME[layer_params.type_layer].c_str());
              INFO(L"Layer[%zu]: Number neuron unit(s): %zu.",
                   layers_params.size() - 1_UZ,
                   layer_params.unit_parameters[0]);

              ++tmp_sub_index;
              // |END| First hidden layer inside the residual block. |END|

              // Second hidden layer inside the residual block.
              layer_params.unit_parameters[0] = static_cast<size_t>(
                  std::max<double>(
                      tmp_widening_factor_units[0],
                      tmp_widening_factor_units[0] + tmp_widening_factors[0]) /
                  2.0);

              layers_params.push_back(layer_params);

              INFO(L"Layer[%zu]: Type: %ls.", layers_params.size() - 1_UZ,
                   LAYER_NAME[layer_params.type_layer].c_str());
              INFO(L"Layer[%zu]: Number neuron unit(s): %zu.",
                   layers_params.size() - 1_UZ,
                   layer_params.unit_parameters[0]);

              ++tmp_sub_index;
              // |END| Second hidden layer inside the residual block. |END|
            }

            tmp_widening_factor_units[0] += tmp_widening_factors[0];
            layer_params.unit_parameters[0] =
                static_cast<size_t>(tmp_widening_factor_units[0]);

            for (; tmp_sub_index != this->residual_block_width;
                 ++tmp_sub_index) {
              layers_params.push_back(layer_params);

              INFO(L"Layer[%zu]: Type: %ls.", layers_params.size() - 1_UZ,
                   LAYER_NAME[layer_params.type_layer].c_str());
              INFO(L"Layer[%zu]: Number neuron unit(s): %zu.",
                   layers_params.size() - 1_UZ,
                   layer_params.unit_parameters[0]);
            }
            // |END| Building block. |END|
          }
          // |END| Residual group #2. |END|

          // Residual group #3.
          //  Pooling layer.
          if (this->use_pooling) {
            layer_params.type_layer = this->type_pooling_layer;
            layer_params.unit_parameters[0] = this->pooling_kernel_size;
            layer_params.unit_parameters[1] = this->pooling_stride;
            layer_params.unit_parameters[2] = this->pooling_padding;
            layer_params.unit_parameters[3] = this->pooling_dilation;
            layer_params.unit_parameters[4] =
                static_cast<size_t>(this->pooling_ceil_mode);
            layers_params.push_back(layer_params);

            INFO(L"Layer[%zu]: Type: %ls.", layers_params.size() - 1_UZ,
                 LAYER_NAME[layer_params.type_layer].c_str());
            INFO(L"Layer[%zu]: Kernel size: %zu.", layers_params.size() - 1_UZ,
                 layer_params.unit_parameters[0]);
            INFO(L"Layer[%zu]: Stride: %zu.", layers_params.size() - 1_UZ,
                 layer_params.unit_parameters[1]);
            INFO(L"Layer[%zu]: Padding: %zu.", layers_params.size() - 1_UZ,
                 layer_params.unit_parameters[2]);
            INFO(L"Layer[%zu]: Dilation: %zu.", layers_params.size() - 1_UZ,
                 layer_params.unit_parameters[3]);
            INFO(L"Layer[%zu]: Ceil mode: %ls.", layers_params.size() - 1_UZ,
                 layer_params.unit_parameters[4] != 0_UZ ? "true" : "false");
          }
          //  |END| Pooling layer. |END|

          for (residual_unit_i = 0_UZ;
               residual_unit_i != n_residual_layers_last_group;
               ++residual_unit_i) {
            // Residual unit.
            layer_params.type_layer = LAYER::RESIDUAL;
            layer_params.unit_parameters[0] = this->residual_block_width;

            layers_params.push_back(layer_params);

            INFO(L"Layer[%zu]: Type: %ls.", layers_params.size() - 1_UZ,
                 LAYER_NAME[layer_params.type_layer].c_str());
            INFO(L"Layer[%zu]: Block depth: %zu.", layers_params.size() - 1_UZ,
                 layer_params.unit_parameters[0]);
            // |END| Residual unit. |END|

            // Building block.
            tmp_sub_index = 0_UZ;
            layer_params.type_layer = this->type_fully_connected;

            if (this->use_bottleneck) {
              // First hidden layer inside the residual block.
              layer_params.unit_parameters[0] =
                  static_cast<size_t>(tmp_widening_factor_units[0]);

              layers_params.push_back(layer_params);

              INFO(L"Layer[%zu]: Type: %ls.", layers_params.size() - 1_UZ,
                   LAYER_NAME[layer_params.type_layer].c_str());
              INFO(L"Layer[%zu]: Number neuron unit(s): %zu.",
                   layers_params.size() - 1_UZ,
                   layer_params.unit_parameters[0]);

              ++tmp_sub_index;
              // |END| First hidden layer inside the residual block. |END|

              // Second hidden layer inside the residual block.
              layer_params.unit_parameters[0] = static_cast<size_t>(
                  std::max<double>(
                      tmp_widening_factor_units[0],
                      tmp_widening_factor_units[0] + tmp_widening_factors[0]) /
                  2.0);

              layers_params.push_back(layer_params);

              INFO(L"Layer[%zu]: Type: %ls.", layers_params.size() - 1_UZ,
                   LAYER_NAME[layer_params.type_layer].c_str());
              INFO(L"Layer[%zu]: Number neuron unit(s): %zu.",
                   layers_params.size() - 1_UZ,
                   layer_params.unit_parameters[0]);

              ++tmp_sub_index;
              // |END| Second hidden layer inside the residual block. |END|
            }

            tmp_widening_factor_units[0] += tmp_widening_factors[0];
            layer_params.unit_parameters[0] =
                static_cast<size_t>(tmp_widening_factor_units[0]);

            for (; tmp_sub_index != this->residual_block_width;
                 ++tmp_sub_index) {
              layers_params.push_back(layer_params);

              INFO(L"Layer[%zu]: Type: %ls.", layers_params.size() - 1_UZ,
                   LAYER_NAME[layer_params.type_layer].c_str());
              INFO(L"Layer[%zu]: Number neuron unit(s): %zu.",
                   layers_params.size() - 1_UZ,
                   layer_params.unit_parameters[0]);
            }
            // |END| Building block. |END|
          }
          // |END| Residual group #3. |END|

          // Last hidden layer.
          layer_params.type_layer = this->type_fully_connected;
          layer_params.unit_parameters[0] =
              static_cast<size_t>(tmp_widening_factor_units[0]);
          layers_params.push_back(layer_params);

          INFO(L"Layer[%zu]: Type: %ls.", layers_params.size() - 1_UZ,
               LAYER_NAME[layer_params.type_layer].c_str());
          INFO(L"Layer[%zu]: Number neuron unit(s): %zu.",
               layers_params.size() - 1_UZ, layer_params.unit_parameters[0]);
          // |END| Last hidden layer. |END|
        } else {
          for (ep = 1_UZ; ep != this->n_layers - 1_UZ; ++ep) {
            layer_params.type_layer = this->type_fully_connected;
            layer_params.unit_parameters[0] = this->N_units[0];
            layers_params.push_back(layer_params);

            INFO(L"Layer[%zu]: Type: %ls.", ep,
                 LAYER_NAME[layer_params.type_layer].c_str());
            INFO(L"Layer[%zu]: Number neuron unit(s): %zu.", ep,
                 layer_params.unit_parameters[0]);
          }
        }
        break;
      case MODEL::RECURRENT:
        for (ep = 1_UZ; ep != this->n_layers - 1_UZ; ++ep) {
          layer_params.use_bidirectional = this->use_bidirectional;
          layer_params.type_layer = LAYER::LSTM;
          layer_params.unit_parameters[0] = this->N_units[0];
          layer_params.unit_parameters[1] = this->N_units[1];
          layers_params.push_back(layer_params);

          INFO(L"Layer[%zu]: Type: %ls.", ep,
               LAYER_NAME[layer_params.type_layer].c_str());

          INFO(L"Layer[%zu]: Number block unit(s): %zu.", ep,
               layer_params.unit_parameters[0]);

          INFO(L"Layer[%zu]: Number cell unit(s) per block: %zu.", ep,
               layer_params.unit_parameters[1]);
        }
        break;
      default:
        ERR(L"Neural network type (%d | %ls) is not managed in "
            L"the switch.",
            this->type_model, MODEL_NAME[this->type_model].c_str());
        delete (model);
        return false;
    }

    layer_params.type_layer = LAYER::FULLY_CONNECTED;
    layer_params.unit_parameters[0] = this->_datasets->get_n_out();
    layers_params.push_back(layer_params);

    INFO(L"Number output(s): %zu.", this->_datasets->get_n_out());

    if (model->compile(layers_params.size(), this->_datasets->get_seq_w(),
                       this->type_model, layers_params.data(),
                       this->allowable_mem) == false) {
      ERR(L"An error has been triggered from the `compile()` function.");
      delete (model);
      return false;
    } else if (model->set_seq_w(this->seq_w) == false) {
      ERR(L"An error has been triggered from the `set_seq_w(%zu)` function.",
          this->seq_w);
      delete (model);
      return false;
    }

    if (this->tied_params) {
      for (ep = 1_UZ; ep != layers_params.size() - 1_UZ; ++ep) {
        if (model->Set__Tied_Parameter(ep, true) == false) {
          ERR(L"An error has been triggered from the "
              L"`Set__Tied_Parameter(%zu, true)` function.",
              ep);
          delete (model);
          return false;
        }
      }
    }

    for (ep = 1_UZ; ep != layers_params.size() - 1_UZ; ++ep) {
      if (model->set_layer_activation_function(ep, this->type_af_hidden) ==
          false) {
        ERR(L"An error has been triggered from the "
            L"`set_layer_activation_function(%zu, %u)` function.",
            ep, this->type_af_hidden);
        delete (model);
        return false;
      }
    }

    if (model->set_layer_activation_function(layers_params.size() - 1_UZ,
                                             this->type_af_output) == false) {
      ERR(L"An error has been triggered from the "
          L"`set_layer_activation_function(%zu, %u)` function.",
          layers_params.size() - 1_UZ, this->type_af_output);
      delete (model);
      return false;
    }

    model->set_loss_fn(this->type_loss_fn);

    model->set_accu_fn(this->type_acc_fn);

    model->set_optimizer(this->type_optimizer);

    switch (this->type_optimizer) {
      case OPTIMIZER::ADABOUND:
      case OPTIMIZER::ADAM:
      case OPTIMIZER::ADAMAX:
      case OPTIMIZER::AMSBOUND:
      case OPTIMIZER::AMSGRAD:
      case OPTIMIZER::NOSADAM:
        model->adam_learning_rate = this->lr;
        break;
      default:
        model->learning_rate = this->lr;
        break;
    }

    model->learning_rate_final = this->lr_final;

    if (this->use_warm_restarts) {
      model->use_warm_restarts = true;
      model->warm_restarts_maximum_learning_rate = this->wm_max_lr;
      model->warm_restarts_minimum_learning_rate = this->wm_min_lr;
      model->warm_restarts_initial_T_i = this->wm_initial_ti;
      model->warm_restarts_multiplier = this->wm_multiplier;
    }

    model->set_clip_gradient(this->clip_gradient);

    if (model->set_clip_gradient(this->clip_gradient_val) == false) {
      ERR(L"An error has been triggered from the "
          L"`set_clip_gradient(%f)` function.",
          this->clip_gradient_val);
      delete (model);
      return false;
    }

    // Regularization L1.
    // TODO: Each layer having a L1 regularization parameter.
    if (model->set_l1(this->l1) == false) {
      ERR(L"An error has been triggered from the `set_l1(%f)` function.",
          this->l1);
      delete (model);
      return false;
    }
    // |END| Regularization L1. |END|

    // Regularization L2.
    // TODO: Each layer having a L2 regularization parameter.
    if (model->set_l2(this->l2) == false) {
      ERR(L"An error has been triggered from the `set_l1(%f)` function.",
          this->l2);
      delete (model);
      return false;
    }
    // |END| Regularization L2. |END|

    // Regularization max-norm constraints.
    // TODO: Each layer having a max-norm constraints regularization
    // parameter.
    if (model->Set__Regularization__Max_Norm_Constraints(this->max_norm) ==
        false) {
      ERR(L"An error has been triggered from the "
          L"`Set__Regularization__Max_Norm_Constraints(%f)` function.",
          this->max_norm);
      delete (model);
      return false;
    }
    // |END| Regularization max-norm constraints. |END|

    // Regularization max-norm constraints.
    // TODO: Each layer having a max-norm constraints regularization
    // parameter.
    if (model->set_weight_decay(this->weight_decay) == false) {
      ERR(L"An error has been triggered from the "
          L"`set_weight_decay(%f)` function.",
          this->weight_decay);
      delete (model);
      return false;
    }
    // |END| Regularization max-norm constraints. |END|

    // Accuracy variance.
    model->Set__Accurancy_Variance(this->acc_var);
    // |END| Accuracy variance. |END|

    // Initialize the weights just at the first run, because sometimes the
    // second initialization differ from the first run.
    if (model_id == 0_UZ) {
      INFO(L"");
      INFO(L"Weight(s) intialization: %ls.",
           INITIALIZER_NAME[this->type_weight_initializer].c_str());

      INFO(L"seed: %u.", this->dist_seed);
      model->real_gen.reset();
      model->gaussian.reset();
      model->real_gen.seed(this->dist_seed);
      model->gaussian.seed(this->dist_seed);

      switch (this->type_weight_initializer) {
        case INITIALIZER::GLOROT_GAUSSIAN:
          model->Initialization__Glorot__Gaussian();
          break;
        case INITIALIZER::GLOROT_UNIFORM:
          model->initialize_weights_with_glorot_uniform();
          break;
        case INITIALIZER::IDENTITY:
          model->Initialization__Identity();
          break;
        case INITIALIZER::LSUV:
          model->Initialize__LSUV();
          break;
        case INITIALIZER::ORTHOGONAL:
          model->layers_initialize_orthogonal();
          break;
        case INITIALIZER::UNIFORM:
          INFO(L"Range: [%f, %f].", this->weight_minval_range,
               this->weight_maxval_range);
          model->real_gen.range(this->weight_minval_range,
                                this->weight_maxval_range);

          model->Initialization__Uniform();
          break;
        default:
          ERR(L"Weights initializer (%d | %ls) is not managed in the switch.",
              this->type_weight_initializer,
              INITIALIZER_NAME[this->type_weight_initializer].c_str());
          return false;
      }

      this->_initial_parameters = new var[model->total_parameters];
      VARCOPY(this->_initial_parameters, model->ptr_array_parameters,
              model->total_parameters * sizeof(var));

      if (this->print_params) {
        for (size_t w = 0_UZ; w != model->total_parameters; ++w)
          INFO(L"W[%zu]: %f", w, cast(this->_initial_parameters[w]));
      }
    } else {
      VARCOPY(model->ptr_array_parameters, this->_initial_parameters,
              model->total_parameters * sizeof(var));
    }

    // Dropout.
    if (this->use_dropout) {
      switch (this->type_dropout) {
        case LAYER_DROPOUT::BERNOULLI:
        case LAYER_DROPOUT::BERNOULLI_INVERTED:
        case LAYER_DROPOUT::GAUSSIAN:
        case LAYER_DROPOUT::UOUT:
          INFO(L"");
          for (ep = 0_UZ; ep != layers_params.size() - 1_UZ; ++ep) {
            if (model->ptr_array_layers[ep].type_layer ==
                    LAYER::FULLY_CONNECTED ||
                model->ptr_array_layers[ep].type_layer ==
                    LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT ||
                model->ptr_array_layers[ep].type_layer ==
                    LAYER::FULLY_CONNECTED_RECURRENT) {
              INFO(L"Layer[%zu], Dropout: %ls(%f).", ep,
                   LAYER_DROPOUT_NAME[this->type_dropout].c_str(),
                   this->dropout_values[0]);
              model->set_dropout(ep, this->type_dropout, this->dropout_values);
            }
          }
          break;
        case LAYER_DROPOUT::SHAKEDROP:
          INFO(L"");
          for (ep = 1_UZ; ep != layers_params.size() - 1_UZ; ++ep) {
            if (model->ptr_array_layers[ep].type_layer == LAYER::RESIDUAL) {
              INFO(L"Layer[%zu], Dropout: %ls(%f).", ep,
                   LAYER_DROPOUT_NAME[this->type_dropout].c_str(),
                   this->dropout_values[0]);
              model->set_dropout(ep, this->type_dropout, this->dropout_values);
            }
          }
          break;
        case LAYER_DROPOUT::ZONEOUT:
          INFO(L"");
          for (ep = 1_UZ; ep != layers_params.size() - 1_UZ; ++ep) {
            if (model->ptr_array_layers[ep].type_layer == LAYER::LSTM ||
                model->ptr_array_layers[ep].type_layer == LAYER::GRU) {
              INFO(L"Layer[%zu], Dropout: %ls(%f, %f).", ep,
                   LAYER_DROPOUT_NAME[this->type_dropout].c_str(),
                   this->dropout_values[0], this->dropout_values[1]);
              model->set_dropout(ep, this->type_dropout, this->dropout_values);
            }
          }
          break;
        default:
          ERR(L"Dropout type (%d | %ls) is not managed in the "
              L"switch. Need to be the fully connected layer or one of its "
              L"variant.",
              this->type_dropout,
              LAYER_DROPOUT_NAME[this->type_dropout].c_str());
          return false;
      }
    }
    // |END| Dropout. |END|

    // Normalization.
    if (this->use_normalization) {
      INFO(L"");
      for (ep = 1_UZ; ep != layers_params.size() - 1u; ++ep) {
        INFO(L"Layer[%zu], Normalization: %ls.", ep,
             LAYER_NORM_NAME[this->type_normalization].c_str());

        model->Set__Layer_Normalization(ep, this->type_normalization);

        model->ptr_array_layers[ep].use_layer_normalization_before_activation =
            this->use_bn_before_af;
      }

      model->Set__Normalization_Momentum_Average(this->bn_mom_avg);

      model->Set__Normalization_Epsilon(this->bn_epsilon);
    }
    // |END| Normalization. |END|

    model->pct_threads = this->pct_threads;

    if (model->set_mp(this->use_mp) == false) {
      ERR(L"An error has been triggered from the `set_mp(%ls)` function.",
          to_wstring(this->use_mp).c_str());
      delete (model);
      return false;
    }
  }

  if (model->set_cu(this->use_cuda, this->_mem_allocate) == false) {
    ERR(L"An error has been triggered from the `set_cu(%ls, %zu)` function.",
        to_wstring(this->use_cuda).c_str(), this->_mem_allocate);
    delete (model);
    return false;
  }

  if (this->use_cuda &&
      model->Initialize__CUDA__Thread(this->_datasets) == false) {
    ERR(L"An error has been triggered from the "
        L"`Initialize__CUDA__Thread(ptr)` function.");
    delete (model);
    return false;
  }

  INFO(L"");
  INFO(L"Total layer(s): %zu", model->total_layers);
  INFO(L"Total basic unit(s): %zu", model->total_basic_units);
  INFO(L"Total neuron unit(s): %zu", model->total_neuron_units);
  INFO(L"Total AF unit(s): %zu", model->total_AF_units);
  INFO(L"Total AF Ind recurrent unit(s): %zu",
       model->total_AF_Ind_recurrent_units);
  INFO(L"Total normalized unit(s): %zu", model->total_normalized_units);
  INFO(L"Total block unit(s): %zu", model->total_block_units);
  INFO(L"Total cell unit(s): %zu", model->total_cell_units);
  INFO(L"Total weight(s): %zu", model->total_weights);
  INFO(L"Total parameter(s): %zu", model->total_parameters);

  // Adept
  if (this->use_adept) {
    INFO(L"");
    INFO(L"Adept: Evaluate %zu example(s) from the testing set.",
        this->_datasets->get_dataset(ENV::TESTG)->DatasetV1::get_n_data());

    time_str = std::chrono::high_resolution_clock::now();

    this->_datasets->Adept__Gradient(model);

    time_end = std::chrono::high_resolution_clock::now();

    INFO(L"Adept: Loss: %.9f.", model->get_loss(ENV::TESTG));
    INFO(L"Adept: Accurancy: %.5f.", model->get_accu(ENV::TESTG));

    tmp_compute_time = static_cast<double>(
                           std::chrono::duration_cast<std::chrono::nanoseconds>(
                               time_end - time_str)
                               .count()) /
                       1e9;

    INFO(L"Adept: Time elapse: %ls",
         Time::time_format(tmp_compute_time).c_str());
  }
  // |END| Adept. |END|

  // Test set.
  if (this->use_infer_set) {
    INFO(L"");
    INFO(L"Evaluate %zu example(s) from the testing set.",
        this->_datasets->get_dataset(ENV::TESTG)->DatasetV1::get_n_data());

    time_str = std::chrono::high_resolution_clock::now();

    this->_datasets->Type_Testing(ENV::TESTG, model);

    time_end = std::chrono::high_resolution_clock::now();

    INFO(L"Loss: %.9f.", model->get_loss(ENV::TESTG));
    INFO(L"Accurancy: %.5f.", model->get_accu(ENV::TESTG));

    tmp_compute_time = static_cast<double>(
                           std::chrono::duration_cast<std::chrono::nanoseconds>(
                               time_end - time_str)
                               .count()) /
                       1e9;

    INFO(L"Time elapse: %ls", Time::time_format(tmp_compute_time).c_str());

    if (this->infer_set_print_x || this->infer_set_print_y) {
      size_t const n_data(
          this->_datasets->get_dataset(ENV::TESTG)->DatasetV1::get_n_data()),
          tmp_number_inputs(
              this->_datasets->get_dataset(ENV::TESTG)->DatasetV1::get_n_inp()),
          tmp_number_outputs(
              this->_datasets->get_dataset(ENV::TESTG)->DatasetV1::get_n_out()),
          seq_w(
              this->_datasets->get_dataset(ENV::TESTG)->DatasetV1::get_seq_w());
      size_t tmp_example_index, tmp_time_step, tmp_input_index;
      for (tmp_example_index = 0_UZ; tmp_example_index != n_data;
           ++tmp_example_index) {
        if (this->infer_set_print_x)
          INFO(L"Data[%zu], Input size: %zu", tmp_example_index,
               tmp_number_inputs);
        if (this->infer_set_print_y)
          INFO(L"Data[%zu], Output size: %zu", tmp_example_index,
               tmp_number_outputs);

        for (tmp_time_step = 0_UZ; tmp_time_step != seq_w; ++tmp_time_step) {
          if (this->infer_set_print_x) {
            if (this->input_image_size == 0_UZ) {
              for (tmp_input_index = 0_UZ; tmp_input_index != tmp_number_inputs;
                   ++tmp_input_index) {
                INFO(L"%f ",
                     this->_datasets->get_inp(
                         tmp_example_index)[tmp_time_step * tmp_number_inputs +
                                            tmp_input_index]);
              }
            } else {
              for (tmp_input_index = 0_UZ; tmp_input_index != tmp_number_inputs;
                   ++tmp_input_index) {
                if (tmp_input_index != 0_UZ &&
                    tmp_input_index % this->input_image_size == 0_UZ) {
                  INFO(L"");
                }

                INFO(L"%.0f ",
                     round(this->_datasets->get_inp(
                         tmp_example_index)[tmp_time_step * tmp_number_inputs +
                                            tmp_input_index]));
              }
            }

            INFO(L"");
          }

          if (this->infer_set_print_y) {
            for (tmp_input_index = 0_UZ; tmp_input_index != tmp_number_outputs;
                 ++tmp_input_index) {
              INFO(L"%f ",
                   cast(model->get_out(tmp_example_index,
                                       tmp_time_step)[tmp_input_index]));
            }

            INFO(L"");
          }
        }

        INFO(L"");
      }
    }

    if (std::isfinite(model->get_loss(ENV::TESTG))) Term::pause();
  }
  // |END| Test set. |END|

  // Valid set.
  if (this->use_valid_set) {
    INFO(L"");
    INFO(L"Evaluate %zu example(s) from the validating set.",
        this->_datasets->get_dataset(ENV::VALID)->DatasetV1::get_n_data());

    time_str = std::chrono::high_resolution_clock::now();

    this->_datasets->Type_Testing(ENV::VALID, model);

    time_end = std::chrono::high_resolution_clock::now();

    INFO(L"Loss: %.9f.", model->get_loss(ENV::VALID));
    INFO(L"Accurancy: %.5f.", model->get_accu(ENV::VALID));

    tmp_compute_time = static_cast<double>(
                           std::chrono::duration_cast<std::chrono::nanoseconds>(
                               time_end - time_str)
                               .count()) /
                       1e9;

    INFO(L"Time elapse: %ls", Time::time_format(tmp_compute_time).c_str());

    if (std::isfinite(model->get_loss(ENV::VALID))) Term::pause();
  }
  // |END| Valid set. |END|

  INFO(L"");
  INFO(L"Optimize %zu example(s) from the training set for %zu epoch(s).",
      this->_datasets->get_dataset(ENV::TRAIN)->DatasetV1::get_n_data(),
      this->n_episodes);

  for (ep = 0_UZ; ep != this->n_episodes; ++ep) {
    // Simulate online training.
    if (this->online_training) {
      INFO(L"");
      INFO(L"Simulate online training");

      real *tmp_ptr_array_inputs, *tmp_ptr_array_outputs;

      tmp_ptr_array_inputs = new real[this->_datasets->get_n_inp()];
      tmp_ptr_array_outputs = new real[this->_datasets->get_n_out()];

      Dist::Real gen_inp;
      for (size_t k(0_UZ); k != this->_datasets->get_n_inp(); ++k) {
        tmp_ptr_array_inputs[k] = gen_inp();
      }

      Dist::Bernoulli gen_out(0.5);
      for (size_t k(0_UZ); k != this->_datasets->get_n_out(); ++k) {
        tmp_ptr_array_outputs[k] = gen_out();
      }

      this->_datasets->push_back(tmp_ptr_array_inputs, tmp_ptr_array_outputs);

      delete[] (tmp_ptr_array_inputs);
      delete[] (tmp_ptr_array_outputs);
    }
    // |END| Simulate online training. |END|

    time_str = std::chrono::high_resolution_clock::now();

    INFO(L"");
    INFO(L"Neural network: Train for %zu sub-epoch(s).", this->n_epochs);

    if (this->use_train_set) {
      for (size_t tmp_sub_epoch_index(0_UZ);
           tmp_sub_epoch_index != this->n_epochs; ++tmp_sub_epoch_index) {
        this->_datasets->train(model);
      }
    }

    time_end = std::chrono::high_resolution_clock::now();

    INFO(L"Neural network: Training loss: %.9f.", model->get_loss(ENV::TRAIN));
    INFO(L"Neural network: Training accuracy: %.5f.",
         model->get_accu(ENV::TRAIN));

    tmp_compute_time = static_cast<double>(
                           std::chrono::duration_cast<std::chrono::nanoseconds>(
                               time_end - time_str)
                               .count()) /
                       1e9;

    INFO(L"[%zu] Time elapse: %ls", ep,
         Time::time_format(tmp_compute_time).c_str());

    tmp_time_total += tmp_compute_time;

    if (std::isfinite(model->get_loss(ENV::TRAIN))) Term::pause();
  }

  INFO(L"");
  INFO(L"Time elapse total: %ls", Time::time_format(tmp_time_total).c_str());

  // Update BN moving average.
  if (this->update_bn_final) {
    INFO(L"");
    INFO(L"Update batch normalization on %zu example(s) "
         L"from the training set.",
        this->_datasets->get_dataset(ENV::TRAIN)->DatasetV1::get_n_data());

    time_str = std::chrono::high_resolution_clock::now();

    std::pair<double, double> const metrics(
        this->_datasets->Type_Update_Batch_Normalization(ENV::TRAIN, model));

    time_end = std::chrono::high_resolution_clock::now();

    INFO(L"Loss: %.9f.", std::get<0>(metrics));
    INFO(L"Accurancy: %.5f.", std::get<1>(metrics));

    tmp_compute_time = static_cast<double>(
                           std::chrono::duration_cast<std::chrono::nanoseconds>(
                               time_end - time_str)
                               .count()) /
                       1e9;

    INFO(L"Time elapse: %ls", Time::time_format(tmp_compute_time).c_str());
  }

  // Test set
  if (this->use_infer_set) {
    INFO(L"");
    INFO(L"Evaluate %zu example(s) from the testing set.",
        this->_datasets->get_dataset(ENV::TESTG)->DatasetV1::get_n_data());

    time_str = std::chrono::high_resolution_clock::now();

    this->_datasets->Type_Testing(ENV::TESTG, model);

    time_end = std::chrono::high_resolution_clock::now();

    INFO(L"Loss: %.9f.", model->get_loss(ENV::TESTG));
    INFO(L"Accurancy: %.5f.", model->get_accu(ENV::TESTG));

    tmp_compute_time = static_cast<double>(
                           std::chrono::duration_cast<std::chrono::nanoseconds>(
                               time_end - time_str)
                               .count()) /
                       1e9;

    INFO(L"Time elapse: %ls", Time::time_format(tmp_compute_time).c_str());

    if (std::isfinite(model->get_loss(ENV::TESTG))) Term::pause();

    if (this->infer_set_print_x || this->infer_set_print_y) {
      size_t const n_data(
          this->_datasets->get_dataset(ENV::TESTG)->DatasetV1::get_n_data()),
          tmp_number_inputs(
              this->_datasets->get_dataset(ENV::TESTG)->DatasetV1::get_n_inp()),
          tmp_number_outputs(
              this->_datasets->get_dataset(ENV::TESTG)->DatasetV1::get_n_out()),
          seq_w(
              this->_datasets->get_dataset(ENV::TESTG)->DatasetV1::get_seq_w());
      size_t tmp_example_index, tmp_time_step, tmp_input_index;
      for (tmp_example_index = 0_UZ; tmp_example_index != n_data;
           ++tmp_example_index) {
        if (this->infer_set_print_x)
          INFO(L"Data[%zu], Input size: %zu", tmp_example_index,
               tmp_number_inputs);
        if (this->infer_set_print_y)
          INFO(L"Data[%zu], Output size: %zu", tmp_example_index,
               tmp_number_outputs);

        for (tmp_time_step = 0_UZ; tmp_time_step != seq_w; ++tmp_time_step) {
          if (this->infer_set_print_x) {
            if (this->input_image_size == 0_UZ) {
              for (tmp_input_index = 0_UZ; tmp_input_index != tmp_number_inputs;
                   ++tmp_input_index) {
                INFO(L"%f ",
                     this->_datasets->get_inp(
                         tmp_example_index)[tmp_time_step * tmp_number_inputs +
                                            tmp_input_index]);
              }
            } else {
              for (tmp_input_index = 0_UZ; tmp_input_index != tmp_number_inputs;
                   ++tmp_input_index) {
                if (tmp_input_index != 0_UZ &&
                    tmp_input_index % this->input_image_size == 0_UZ) {
                  INFO(L"");
                }

                INFO(L"%.0f ",
                     round(this->_datasets->get_inp(
                         tmp_example_index)[tmp_time_step * tmp_number_inputs +
                                            tmp_input_index]));
              }
            }

            INFO(L"");
          }

          if (this->infer_set_print_y) {
            for (tmp_input_index = 0_UZ; tmp_input_index != tmp_number_outputs;
                 ++tmp_input_index) {
              INFO(L"%f ",
                   cast(model->get_out(tmp_example_index,
                                       tmp_time_step)[tmp_input_index]));
            }

            INFO(L"");
          }
        }

        INFO(L"");
      }
    }
  }
  // |END| Test set |END|

  // Valid set
  if (this->use_valid_set) {
    INFO(L"");
    INFO(L"Evaluate %zu example(s) from the validating set.",
        this->_datasets->get_dataset(ENV::VALID)->DatasetV1::get_n_data());

    time_str = std::chrono::high_resolution_clock::now();

    this->_datasets->Type_Testing(ENV::VALID, model);

    time_end = std::chrono::high_resolution_clock::now();

    INFO(L"Loss: %.9f.", model->get_loss(ENV::VALID));
    INFO(L"Accurancy: %.5f.", model->get_accu(ENV::VALID));

    tmp_compute_time = static_cast<double>(
                           std::chrono::duration_cast<std::chrono::nanoseconds>(
                               time_end - time_str)
                               .count()) /
                       1e9;

    INFO(L"Time elapse: %ls", Time::time_format(tmp_compute_time).c_str());
  }
  // |END| Valid set |END|

  if (this->copy_params) {
    INFO(L"");
    INFO(L"Neural network: copy.");

    Model *model_trained(new Model);

    if (model_trained->copy(*model) == false)
      ERR(L"An error has been triggered from the `copy(ptr)` function.");

    // Test set
    if (this->use_infer_set) {
      INFO(L"");
      INFO(L"Evaluate %zu example(s) from the testing set.",
           this->_datasets->get_dataset(ENV::TESTG)->DatasetV1::get_n_data());

      time_str = std::chrono::high_resolution_clock::now();

      this->_datasets->Type_Testing(ENV::TESTG, model_trained);

      time_end = std::chrono::high_resolution_clock::now();

      INFO(L"Loss: %.9f.", model_trained->get_loss(ENV::TESTG));
      INFO(L"Accurancy: %.5f.", model_trained->get_accu(ENV::TESTG));

      tmp_compute_time =
          static_cast<double>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(time_end -
                                                                   time_str)
                  .count()) /
          1e9;

      INFO(L"Time elapse: %ls", Time::time_format(tmp_compute_time).c_str());

      if (std::isfinite(model_trained->get_loss(ENV::TESTG))) Term::pause();
    }
    // |END| Test set |END|

    // Valid set
    if (this->use_valid_set) {
      INFO(L"");
      INFO(L"Evaluate %zu example(s) from the validating set.",
          this->_datasets->get_dataset(ENV::VALID)->DatasetV1::get_n_data());

      time_str = std::chrono::high_resolution_clock::now();

      this->_datasets->Type_Testing(ENV::VALID, model_trained);

      time_end = std::chrono::high_resolution_clock::now();

      INFO(L"Loss: %.9f.", model_trained->get_loss(ENV::VALID));
      INFO(L"Accurancy: %.5f.", model_trained->get_accu(ENV::VALID));

      tmp_compute_time =
          static_cast<double>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(time_end -
                                                                   time_str)
                  .count()) /
          1e9;

      INFO(L"Time elapse: %ls", Time::time_format(tmp_compute_time).c_str());

      if (std::isfinite(model_trained->get_loss(ENV::VALID))) Term::pause();
    }
    // |END| Valid set |END|

    delete (model_trained);
  }

  if (this->save_params) {
    INFO(L"");
    INFO(L"save params to `%ls`.", path_params.c_str());
    if (model->save_params(path_params) == false)
      ERR(L"A error has been return while saving dimension parameters to "
          L"`%ls`.",
          path_params.c_str());

    INFO(L"save spec params to `%ls`.", path_spec_params.c_str());
    if (model->save_spec_params(path_spec_params) == false)
      ERR(L"A error has been return while saving general parameters to "
          L"`%ls`.",
          path_spec_params.c_str());

    INFO(L"%ls", model->Get__Parameters().c_str());
  }

  if (model_id == 0_UZ) {
    this->_past_loss = model->get_loss(ENV::TESTG);
  } else if (model->get_loss(ENV::TESTG) != this->_past_loss) {
    ERR(L"%.9f != %.9f", model->get_loss(ENV::TESTG), this->_past_loss);
    delete (model);
    return false;
  }

  INFO(L"Neural network: Deallocate.");
  delete (model);

  return true;
}

bool Experimental::check_attributes(void) {
  if (this->wf_alphas[0] <= -static_cast<long long int>(this->N_units[0])) {
    ERR(L"Widening factor, alpha[0] (%lld) can not be less or "
        L"equal to -%zu. At line %d.",
        this->wf_alphas[0], this->N_units[0]);
    return false;
  } else if (this->wf_alphas[1] <=
             -static_cast<long long int>(this->N_units[1])) {
    ERR(L"Widening factor, alpha[1] (%lld) can not be less or "
        L"equal to -%zu. At line %d.",
        this->wf_alphas[1], this->N_units[1]);
    return false;
  } else if (this->residual_block_width < 2_UZ) {
    ERR(L"A residual block must have a depth of at least 2.");
    return false;
  } else if (this->use_bottleneck && this->residual_block_width == 2_UZ) {
    ERR(L"Can not use a residual bottleneck with a block depth of 2.");
    return false;
  }

  switch (this->type_fully_connected) {
    case LAYER::FULLY_CONNECTED:
    case LAYER::FULLY_CONNECTED_RECURRENT:
    case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
      break;
    default:
      ERR(L"The `switch` require a dense layer or one of its variant. "
          L"Not a (%d | %ls).",
          this->type_fully_connected,
          LAYER_NAME[this->type_fully_connected].c_str());
      return false;
  }

  switch (this->type_pooling_layer) {
    case LAYER::AVERAGE_POOLING:
    case LAYER::MAX_POOLING:
      break;
    default:
      ERR(L"The `switch` require a pooling layer. Not a (%d | %ls).",
          this->type_pooling_layer,
          LAYER_NAME[this->type_pooling_layer].c_str());
      return false;
  }

  return true;
}

bool Experimental::setup_datasets(void) {
  DATASET_FORMAT::TYPE tmp_type_dataset_file;

  INFO(L"");
  INFO(L"DatasetV1: Initialization.");
  if (scan_datasets(tmp_type_dataset_file, this->dset_path_name) == false) {
    ERR(L"An error has been triggered from the `scan_datasets()` function.");
    return false;
  }

  this->_datasets = new Datasets(tmp_type_dataset_file, this->dset_path_name);

  DatasetsParams datasets_params;
  datasets_params.type_storage = 0;
  datasets_params.type_train = 0;

  if (this->_datasets->Preparing_Dataset_Manager(&datasets_params) == false) {
    ERR(L"An error has been triggered from the "
        L"`Preparing_Dataset_Manager(ptr)` function.");
    return false;
  }

  if (this->online_training && this->_datasets->Set__Maximum_Data(
                                   this->_datasets->get_n_data()) == false) {
    ERR(L"An error has been triggered from the "
        L"`Set__Maximum_Data(%zu)` function.",
        this->_datasets->get_n_data());
    return false;
  }

  return true;
}

[[deprecated("Untested.")]] void run_custom(void) {
#ifdef _WIN32
  if (SetConsoleTitleW(L"RUN-CUSTOM") == FALSE)
    WARN(L"Couldn't set a title to the console.");
#endif

  // nvvp: NVIDIA Visual Profiler
  // nsight: NVIDIA Nsight™ Graphics
  bool const nvvp_or_nsight(
      Term::accept(L"Are you debugging from NVVP/Nsight?"));

  Experimental experimental(nvvp_or_nsight);

  if (experimental.check_attributes() == false) {
    ERR(L"An error has been triggered from the "
        L"`check_attributes` function.");
    return;
  }

  experimental.print_attributes();

#ifdef COMPILE_CUDA
  if (experimental.this->use_cuda) {
#ifdef _WIN32
    if (SetConsoleTitleW(L"RUN-CUSTOM [CUDA ENABLED]") == FALSE)
      WARN(L"Couldn't set a title to the console.");
#endif

    experimental.setup_cuda();
    experimental.print_cudevice()
  }
#endif  // COMPILE_CUDA

  if (experimental.setup_datasets() == false) {
    ERR(L"An error has been triggered from the "
        L"`setup_datasets` function.");
    return;
  }

#ifdef COMPILE_CUDA
  if (experimental.setup_cudevice() == false) {
    ERR(L"An error has been triggered from the "
        L"`setup_cudevice` function.");
    return;
  }
#endif  // COMPILE_CUDA

  experimental.start();

  if (nvvp_or_nsight == false) Term::pause();
}
