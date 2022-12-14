/* Copyright 2022 Sébastien Kéroack. All Rights Reserved.

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

// File header:
#include "run/deep-learning/mnist.hpp"

// Deep learning:
#include "deep-learning/data/dataset/mnist.hpp"
#include "deep-learning/data/string.hpp"
#include "deep-learning/io/flags.hpp"
#include "deep-learning/io/logger.hpp"
#include "deep-learning/nn/learner.hpp"
#include "deep-learning/v1/learner/models.hpp"

// Standard:
#ifdef _WIN32
#include <windows.h>
#endif

using namespace DL::Str;

namespace DL {
constexpr size_t ALLOWABLE_MODEL_BYTES = 512_UZ * ::MEGABYTE;

class MNISTLearner : public Learner {
 protected:
  void _create_model(void) {
    std::vector<int> N_units = {128};

    this->model = new v1::Model;

    v1::Layer_Parameters layer_params;
    std::vector<v1::Layer_Parameters> layers_params;
    layer_params.type_layer = v1::LAYER::FULLY_CONNECTED;
    layer_params.unit_parameters[0] = this->datasets[ENV::TRAIN]->n_inp;
    layers_params.push_back(layer_params);

    for (size_t i(0_UZ); i != N_units.size(); ++i) {
      layer_params.type_layer = v1::LAYER::FULLY_CONNECTED;
      layer_params.unit_parameters[0] = static_cast<size_t>(N_units[i]);
      layers_params.push_back(layer_params);
    }

    layer_params.type_layer = v1::LAYER::FULLY_CONNECTED;
    layer_params.unit_parameters[0] = this->datasets[ENV::TRAIN]->n_out;
    layers_params.push_back(layer_params);

    this->model->compile(layers_params.size(), 1_UZ, v1::MODEL::FEEDFORWARD,
                         layers_params.data(), ALLOWABLE_MODEL_BYTES);

    for (size_t i(1_UZ); i != layers_params.size() - 1_UZ; ++i)
      this->model->set_layer_activation_function(i, v1::ACTIVATION::RELU);
    this->model->set_layer_activation_function(layers_params.size() - 1_UZ,
                                               v1::ACTIVATION::SOFTMAX);

    this->model->initialize_weights_with_glorot_uniform();
    this->model->set_loss_fn(v1::LOSS_FN::CROSS_ENTROPY);
    this->model->set_accu_fn(v1::ACCU_FN::CROSS_ENTROPY);
    this->model->set_optimizer(v1::OPTIMIZER::ADAM);
    this->model->adam_learning_rate = 1e-3_r;

    this->model->set_max_batch_size(128_UZ);
    this->model->set_mp(true);
  }

  void _create_datasets(void) {
    for (auto const &env : {ENV::TRAIN, ENV::VALID}) {
      this->datasets.push_back(std::make_unique<MNIST>(this->dirs->datasets));
      if (this->datasets[env]->load(env) == false)
        throw std::runtime_error(
            "An error has been triggered from the `MNIST::load()` function.");
    }

    for (size_t i(0_UZ); i != 3; ++i)
      this->datasets[ENV::TRAIN]->print_sample(i);
  }

 public:
  MNISTLearner(std::wstring const &name, int n_iters)
      : Learner(name, n_iters){};
};

bool run_mnist(void) {
#ifdef _WIN32
  if (SetConsoleTitleW(L"MNIST") == FALSE)
    WARN(L"Couldn't set a title to the console.");
#endif

  MNISTLearner learner(L"mnist", static_cast<int>(flags->get(L"n_iters", 0.0)));

  try {
    learner.initialize();
    learner.optimize();
  } catch (std::exception const &err) {
    ERR(L"Caught exception: %ls", to_wstring(err.what()).c_str());
  }

  return true;
}
}  // namespace DL
