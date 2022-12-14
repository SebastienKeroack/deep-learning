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

#include "deep-learning/v1/learner/model.hpp"
#include "deep-learning/data/string.hpp"
#include "deep-learning/data/time.hpp"
#include "deep-learning/io/logger.hpp"
#include "deep-learning/v1/mem/reallocate.hpp"

// `CRTDBG_NEW` is not compatible with `Eigen`.
#ifdef _CRTDBG_MAP_ALLOC
#undef new
#endif

#include <eigen3/Eigen/QR>

#ifdef _CRTDBG_MAP_ALLOC
#define new CRTDBG_NEW
#endif

using Index = Eigen::Index;
using MatrixXT = Eigen::Matrix<real, -1, -1>;

namespace DL::v1 {
void Model::layers_initialize_orthogonal(bool const pre_initialize,
                                         real const bias) {
  Layer const *const last_layer(this->ptr_last_layer);
  Layer *layer_it(this->ptr_array_layers + 1);

  // Loop though each layer.
  for (; layer_it != last_layer; ++layer_it) {
    // If the current layer is a pooling/residual layer, continue.
    if (layer_it->type_layer == LAYER::AVERAGE_POOLING ||
        layer_it->type_layer == LAYER::MAX_POOLING ||
        layer_it->type_layer == LAYER::RESIDUAL)
      continue;

    switch (layer_it->type_layer) {
      case LAYER::FULLY_CONNECTED:
        this->weights_initialize_orthogonal(
            *layer_it->ptr_array_neuron_units->ptr_number_connections,
            static_cast<size_t>(layer_it->ptr_last_neuron_unit -
                                layer_it->ptr_array_neuron_units),
            this->Initialization__Gain__Scale(
                *layer_it->ptr_array_AF_units->ptr_type_activation_function),
            this->ptr_array_parameters + *layer_it->ptr_first_connection_index);

        this->layer_initialize_const_bias(bias, layer_it);
        break;
      case LAYER::FULLY_CONNECTED_INDEPENDENTLY_RECURRENT:
        this->weights_initialize_orthogonal(
            *layer_it->ptr_array_neuron_units->ptr_number_connections,
            static_cast<size_t>(layer_it->ptr_last_neuron_unit -
                                layer_it->ptr_array_neuron_units),
            this->Initialization__Gain__Scale(
                *layer_it->ptr_array_AF_Ind_recurrent_units
                     ->ptr_type_activation_function),
            this->ptr_array_parameters + *layer_it->ptr_first_connection_index);

        this->indrec_initialize_uniform(layer_it);

        this->layer_initialize_const_bias(bias, layer_it);
        break;
      case LAYER::LSTM:
        this->lstm_initialize_orthogonal(layer_it);

        this->lstm_initialize_const_bias(bias, layer_it);
        break;
      default:
        ERR(L"Can not initialize weights in the layer %zu with (%d | %ls) as "
            L"the type layer.",
            static_cast<size_t>(layer_it - this->ptr_array_layers),
            layer_it->type_layer, LAYER_NAME[layer_it->type_layer].c_str());
        break;
    }
  }

  if (pre_initialize == false) {
    // Independently recurrent neural network.
    if (this->seq_w > 1_UZ && this->n_time_delay + 1_UZ == this->seq_w)
      this->indrec_initialize_uniform_ltm();

    if (this->ptr_array_derivatives_parameters != nullptr)
      this->clear_training_arrays();

    if (this->Use__Normalization())
      this->Clear__Parameter__Normalized_Unit();

    this->_initialized__weight = true;
    this->_type_weights_initializer = INITIALIZER::ORTHOGONAL;
  }
}

MatrixXT diag_part(MatrixXT out) {
  MatrixXT diag(out.diagonal());

  out.setZero();

  Index const smallest_length(out.rows() < out.cols() ? out.rows()
                                                      : out.cols());

  for (Index i(0); i != smallest_length; ++i) out(i, i) = diag(i);

  return out;
}

[[deprecated("Not properly implemented.")]] void
Model::lstm_initialize_orthogonal(Layer const *const layer_it) {
  // NotImplementedError.
  // ...
  // ...
}

void Model::weights_initialize_orthogonal(size_t const rows, size_t const cols,
                                          real const scale, var *weights) {
  this->weights_initialize_gaussian(weights, weights + rows * cols, 1_r);

  long const minval(static_cast<long>(std::min(rows, cols)));
  long const maxval(static_cast<long>(std::max(rows, cols)));
  size_t const length(rows * cols);

#if DEEPLEARNING_USE_ADEPT
  real *weights_(new real[rows * cols]);
  Mem::copy(weights, weights + length, weights_);
  MatrixXT W(Eigen::Map<MatrixXT>(weights_, maxval, minval));
  delete[](weights_);
#else
  MatrixXT W(Eigen::Map<MatrixXT>(weights, maxval, minval));
#endif

  Eigen::ColPivHouseholderQR<Eigen::DenseBase<MatrixXT>::PlainMatrix> QR(
      W.colPivHouseholderQr());
  W = QR.matrixQ() * diag_part(QR.matrixR()).cwiseSign();
  if (rows >= cols) W.transposeInPlace();
  if (scale != 1_r) W *= scale;

  Mem::copy(W.data(), W.data() + length, weights);
}
}  // namespace DL
