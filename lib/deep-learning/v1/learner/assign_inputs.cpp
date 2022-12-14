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
#include "deep-learning/v1/mem/reallocate.hpp"

namespace DL::v1 {
void Model::assign_inputs_fwp_st(size_t const batch_size,
                                 real const *const *const Xm) {
  Layer *const layer(this->ptr_array_layers);

  AF_unit *const unit_it(layer->ptr_array_AF_units);

  size_t const n_inp(static_cast<size_t>(layer->ptr_last_AF_unit - unit_it));
  size_t i;

  // Loop through each sample data.
  for (i = 0_UZ; i != batch_size; ++i)
    Mem::copy(Xm[i], Xm[i] + n_inp, unit_it->ptr_array_values + i * n_inp);

  if (this->type_state_propagation == PROPAGATION::TRAINING) {
    switch (layer->type_dropout) {
      case LAYER_DROPOUT::BERNOULLI:
        this->Forward_Pass__Dropout__Bernoulli__Training__Loop(
            layer->ptr_array__mask__dropout__bernoulli, 0_UZ, batch_size, n_inp,
            layer->ptr_array_AF_units->ptr_array_values);
        break;
      case LAYER_DROPOUT::BERNOULLI_INVERTED:
        this->Forward_Pass__Dropout__Bernoulli__Inverted__Loop(
            layer->ptr_array__mask__dropout__bernoulli, 0_UZ, batch_size, n_inp,
            layer->dropout_values[0] == 0_r ? 0_r
                                            : 1_r / layer->dropout_values[0],
            layer->ptr_array_AF_units->ptr_array_values);
        break;
      case LAYER_DROPOUT::GAUSSIAN:
        this->Forward_Pass__Dropout__Gaussian__Loop(
            0_UZ, batch_size, n_inp, layer->dropout_values[0],
            layer->ptr_array_AF_units->ptr_array_values);
        break;
      case LAYER_DROPOUT::UOUT:
        this->Forward_Pass__Dropout__Uout__Loop(
            0_UZ, batch_size, n_inp, layer->dropout_values[0],
            layer->ptr_array_AF_units->ptr_array_values);
        break;
      // TODO: Alpha dropout forward pass.
      default:
        break;
    }
  } else {
    switch (layer->type_dropout) {
      case LAYER_DROPOUT::BERNOULLI:
        this->Forward_Pass__Dropout__Bernoulli__Inference__Loop(
            0_UZ, batch_size, n_inp, layer->dropout_values[0],
            layer->ptr_array_AF_units->ptr_array_values);
        break;
      default:
        break;
    }
  }
}

void Model::assign_inputs_fwp_mp(size_t const batch_size,
                                 real const *const *const Xm) {
  int const batch_size_(static_cast<int>(batch_size));
  int i;

  Layer *const layer(this->ptr_array_layers);

  AF_unit *const unit_it(layer->ptr_array_AF_units);

  size_t const n_inp(static_cast<size_t>(layer->ptr_last_AF_unit - unit_it));

  // Loop through each sample data.
#pragma omp for schedule(static)
  for (i = 0; i < batch_size_; ++i)
    Mem::copy(Xm[i], Xm[i] + n_inp,
              unit_it->ptr_array_values + static_cast<size_t>(i) * n_inp);

  if (this->type_state_propagation == PROPAGATION::TRAINING) {
    switch (layer->type_dropout) {
      case LAYER_DROPOUT::BERNOULLI:
        this->Forward_Pass__Dropout__Bernoulli__Training__OpenMP(
            layer->ptr_array__mask__dropout__bernoulli, 0_UZ, batch_size, n_inp,
            layer->ptr_array_AF_units->ptr_array_values);
        break;
      case LAYER_DROPOUT::BERNOULLI_INVERTED:
        this->Forward_Pass__Dropout__Bernoulli__Inverted__OpenMP(
            layer->ptr_array__mask__dropout__bernoulli, 0_UZ, batch_size, n_inp,
            layer->dropout_values[0] == 0_r ? 0_r
                                            : 1_r / layer->dropout_values[0],
            layer->ptr_array_AF_units->ptr_array_values);
        break;
      case LAYER_DROPOUT::GAUSSIAN:
        this->Forward_Pass__Dropout__Gaussian__OpenMP(
            0_UZ, batch_size, n_inp, layer->dropout_values[0],
            layer->ptr_array_AF_units->ptr_array_values);
        break;
      case LAYER_DROPOUT::UOUT:
        this->Forward_Pass__Dropout__Uout__OpenMP(
            0_UZ, batch_size, n_inp, layer->dropout_values[0],
            layer->ptr_array_AF_units->ptr_array_values);
        break;
      default:
        break;
    }
  } else {
    switch (layer->type_dropout) {
      case LAYER_DROPOUT::BERNOULLI:
        this->Forward_Pass__Dropout__Bernoulli__Inference__OpenMP(
            0_UZ, batch_size, n_inp, layer->dropout_values[0],
            layer->ptr_array_AF_units->ptr_array_values);
        break;
      default:
        break;
    }
  }
}

void Model::assign_inputs_rec_st(size_t const batch_size,
                                 real const *const *const Xm) {
  Layer *const layer(this->ptr_array_layers);

  AF_unit *const unit_it(layer->ptr_array_AF_units);

  size_t const n_inp(static_cast<size_t>(layer->ptr_last_AF_unit - unit_it));
  size_t i, t;

  // Loop through each sample data.
  for (i = 0_UZ; i != batch_size; ++i)
    for (t = 0_UZ; t != this->seq_w; ++t)
      Mem::copy(
          Xm[i] + t * n_inp, Xm[i] + t * n_inp + n_inp,
          unit_it->ptr_array_values + i * n_inp + this->batch_size * n_inp * t);

  if (this->type_state_propagation == PROPAGATION::TRAINING) {
    switch (layer->type_dropout) {
      case LAYER_DROPOUT::BERNOULLI:
        for (t = 0_UZ; t != this->seq_w; ++t) {
          this->Forward_Pass__Dropout__Bernoulli__Training__Loop(
              layer->ptr_array__mask__dropout__bernoulli, t, batch_size, n_inp,
              unit_it->ptr_array_values);
        }
        break;
      case LAYER_DROPOUT::BERNOULLI_INVERTED:
        for (t = 0_UZ; t != this->seq_w; ++t) {
          this->Forward_Pass__Dropout__Bernoulli__Inverted__Loop(
              layer->ptr_array__mask__dropout__bernoulli, t, batch_size, n_inp,
              layer->dropout_values[0] == 0_r ? 0_r
                                              : 1_r / layer->dropout_values[0],
              unit_it->ptr_array_values);
        }
        break;
      case LAYER_DROPOUT::GAUSSIAN:
        for (t = 0_UZ; t != this->seq_w; ++t) {
          this->Forward_Pass__Dropout__Gaussian__Loop(
              t, batch_size, n_inp, layer->dropout_values[0],
              unit_it->ptr_array_values);
        }
        break;
      case LAYER_DROPOUT::UOUT:
        for (t = 0_UZ; t != this->seq_w; ++t) {
          this->Forward_Pass__Dropout__Uout__Loop(t, batch_size, n_inp,
                                                  layer->dropout_values[0],
                                                  unit_it->ptr_array_values);
        }
        break;
      default:
        break;
    }
  } else {
    switch (layer->type_dropout) {
      case LAYER_DROPOUT::BERNOULLI:
        for (t = 0_UZ; t != this->seq_w; ++t) {
          this->Forward_Pass__Dropout__Bernoulli__Inference__Loop(
              t, batch_size, n_inp, layer->dropout_values[0],
              unit_it->ptr_array_values);
        }
        break;
      default:
        break;
    }
  }
}

void Model::assign_inputs_rec_mp(size_t const batch_size,
                                 real const *const *const Xm) {
  int const batch_size_(static_cast<int>(batch_size));
  int i;

  Layer *const layer(this->ptr_array_layers);

  AF_unit *const unit_it(layer->ptr_array_AF_units);

  size_t const n_inp(static_cast<size_t>(layer->ptr_last_AF_unit - unit_it));
  size_t t;

  // Loop through each sample data.
#pragma omp for schedule(static)
  for (i = 0; i < batch_size_; ++i)
    for (t = 0_UZ; t != this->seq_w; ++t)
      Mem::copy(
          Xm[i] + t * n_inp, Xm[i] + t * n_inp + n_inp,
          unit_it->ptr_array_values + static_cast<size_t>(i) * n_inp + this->batch_size * n_inp * t);

  if (this->type_state_propagation == PROPAGATION::TRAINING) {
    switch (layer->type_dropout) {
      case LAYER_DROPOUT::BERNOULLI:
        for (t = 0_UZ; t != this->seq_w; ++t) {
          this->Forward_Pass__Dropout__Bernoulli__Training__OpenMP(
              layer->ptr_array__mask__dropout__bernoulli, t, batch_size, n_inp,
              unit_it->ptr_array_values);
        }
        break;
      case LAYER_DROPOUT::BERNOULLI_INVERTED:
        for (t = 0_UZ; t != this->seq_w; ++t) {
          this->Forward_Pass__Dropout__Bernoulli__Inverted__OpenMP(
              layer->ptr_array__mask__dropout__bernoulli, t, batch_size, n_inp,
              layer->dropout_values[0] == 0_r ? 0_r
                                              : 1_r / layer->dropout_values[0],
              unit_it->ptr_array_values);
        }
        break;
      case LAYER_DROPOUT::GAUSSIAN:
        for (t = 0_UZ; t != this->seq_w; ++t) {
          this->Forward_Pass__Dropout__Gaussian__OpenMP(
              t, batch_size, n_inp, layer->dropout_values[0],
              unit_it->ptr_array_values);
        }
        break;
      case LAYER_DROPOUT::UOUT:
        for (t = 0_UZ; t != this->seq_w; ++t) {
          this->Forward_Pass__Dropout__Uout__OpenMP(t, batch_size, n_inp,
                                                    layer->dropout_values[0],
                                                    unit_it->ptr_array_values);
        }
        break;
      default:
        break;
    }
  } else {
    switch (layer->type_dropout) {
      case LAYER_DROPOUT::BERNOULLI:
        for (t = 0_UZ; t != this->seq_w; ++t) {
          this->Forward_Pass__Dropout__Bernoulli__Inference__OpenMP(
              t, batch_size, n_inp, layer->dropout_values[0],
              unit_it->ptr_array_values);
        }
        break;
      default:
        break;
    }
  }
}

void Model::assign_inputs_pre_train_fwp_st(size_t const batch_size,
                                           real const *const *const Xm) {
  Layer *const layer(this->ptr_array_layers);

  AF_unit *const unit_it(layer->ptr_array_AF_units);

  size_t const n_inp(static_cast<size_t>(layer->ptr_last_AF_unit - unit_it));
  size_t i;

  // Loop through each sample data.
  for (i = 0_UZ; i != batch_size; ++i)
    Mem::copy(Xm[i], Xm[i] + n_inp, unit_it->ptr_array_values + i * n_inp);

  if (this->type_state_propagation == PROPAGATION::TRAINING &&
      this->pre_training_level == 1_UZ) {
    switch (layer->type_dropout) {
      case LAYER_DROPOUT::BERNOULLI:
        this->Forward_Pass__Dropout__Bernoulli__Training__Loop(
            layer->ptr_array__mask__dropout__bernoulli, 0_UZ, batch_size, n_inp,
            unit_it->ptr_array_values);
        break;
      case LAYER_DROPOUT::BERNOULLI_INVERTED:
        this->Forward_Pass__Dropout__Bernoulli__Inverted__Loop(
            layer->ptr_array__mask__dropout__bernoulli, 0_UZ, batch_size, n_inp,
            layer->dropout_values[0] == 0_r ? 0_r
                                            : 1_r / layer->dropout_values[0],
            unit_it->ptr_array_values);
        break;
      case LAYER_DROPOUT::GAUSSIAN:
        this->Forward_Pass__Dropout__Gaussian__Loop(0_UZ, batch_size, n_inp,
                                                    layer->dropout_values[0],
                                                    unit_it->ptr_array_values);
        break;
      case LAYER_DROPOUT::UOUT:
        this->Forward_Pass__Dropout__Uout__Loop(0_UZ, batch_size, n_inp,
                                                layer->dropout_values[0],
                                                unit_it->ptr_array_values);
        break;
      default:
        break;
    }
  } else {
    switch (layer->type_dropout) {
      case LAYER_DROPOUT::BERNOULLI:
        this->Forward_Pass__Dropout__Bernoulli__Inference__Loop(
            0_UZ, batch_size, n_inp, layer->dropout_values[0],
            unit_it->ptr_array_values);
        break;
      default:
        break;
    }
  }
}

void Model::assign_inputs_pre_train_fwp_mp(size_t const batch_size,
                                           real const *const *const Xm) {
  int const batch_size_(static_cast<int>(batch_size));
  int i;

  Layer *const layer(this->ptr_array_layers);

  AF_unit *const unit_it(layer->ptr_array_AF_units);

  size_t const n_inp(static_cast<size_t>(layer->ptr_last_AF_unit - unit_it));

  // Loop through each sample data.
#pragma omp for schedule(static)
  for (i = 0; i < batch_size_; ++i)
    Mem::copy(Xm[i], Xm[i] + n_inp,
              unit_it->ptr_array_values + static_cast<size_t>(i) * n_inp);

  if (this->type_state_propagation == PROPAGATION::TRAINING &&
      this->pre_training_level == 1_UZ) {
    switch (layer->type_dropout) {
      case LAYER_DROPOUT::BERNOULLI:
        this->Forward_Pass__Dropout__Bernoulli__Training__OpenMP(
            layer->ptr_array__mask__dropout__bernoulli, 0_UZ, batch_size, n_inp,
            unit_it->ptr_array_values);
        break;
      case LAYER_DROPOUT::BERNOULLI_INVERTED:
        this->Forward_Pass__Dropout__Bernoulli__Inverted__OpenMP(
            layer->ptr_array__mask__dropout__bernoulli, 0_UZ, batch_size, n_inp,
            layer->dropout_values[0] == 0_r ? 0_r
                                            : 1_r / layer->dropout_values[0],
            unit_it->ptr_array_values);
        break;
      case LAYER_DROPOUT::GAUSSIAN:
        this->Forward_Pass__Dropout__Gaussian__OpenMP(
            0_UZ, batch_size, n_inp, layer->dropout_values[0],
            unit_it->ptr_array_values);
        break;
      case LAYER_DROPOUT::UOUT:
        this->Forward_Pass__Dropout__Uout__OpenMP(0_UZ, batch_size, n_inp,
                                                  layer->dropout_values[0],
                                                  unit_it->ptr_array_values);
        break;
      default:
        break;
    }
  } else {
    switch (layer->type_dropout) {
      case LAYER_DROPOUT::BERNOULLI:
        this->Forward_Pass__Dropout__Bernoulli__Inference__OpenMP(
            0_UZ, batch_size, n_inp, layer->dropout_values[0],
            unit_it->ptr_array_values);
        break;
      default:
        break;
    }
  }
}

void Model::assign_inputs_pre_train_rec_st(size_t const batch_size,
                                           real const *const *const Xm) {
  Layer *const layer(this->ptr_array_layers);

  AF_unit *const unit_it(layer->ptr_array_AF_units);

  size_t const n_inp(static_cast<size_t>(layer->ptr_last_AF_unit - unit_it));
  size_t i, t;

  // Loop through each sample data.
  for (i = 0_UZ; i != batch_size; ++i)
    for (t = 0_UZ; t != this->seq_w; ++t)
      Mem::copy(Xm[i] + t * n_inp, Xm[i] + t * n_inp + n_inp,
                unit_it->ptr_array_values + static_cast<size_t>(i) * n_inp +
                    this->batch_size * n_inp * t);

  if (this->type_state_propagation == PROPAGATION::TRAINING &&
      this->pre_training_level == 1_UZ) {
    switch (layer->type_dropout) {
      case LAYER_DROPOUT::BERNOULLI:
        for (t = 0_UZ; t != this->seq_w; ++t) {
          this->Forward_Pass__Dropout__Bernoulli__Training__Loop(
              layer->ptr_array__mask__dropout__bernoulli, t, batch_size, n_inp,
              unit_it->ptr_array_values);
        }
        break;
      case LAYER_DROPOUT::BERNOULLI_INVERTED:
        for (t = 0_UZ; t != this->seq_w; ++t) {
          this->Forward_Pass__Dropout__Bernoulli__Inverted__Loop(
              layer->ptr_array__mask__dropout__bernoulli, t, batch_size, n_inp,
              layer->dropout_values[0] == 0_r ? 0_r
                                              : 1_r / layer->dropout_values[0],
              unit_it->ptr_array_values);
        }
        break;
      case LAYER_DROPOUT::GAUSSIAN:
        for (t = 0_UZ; t != this->seq_w; ++t) {
          this->Forward_Pass__Dropout__Gaussian__Loop(
              t, batch_size, n_inp, layer->dropout_values[0],
              unit_it->ptr_array_values);
        }
        break;
      case LAYER_DROPOUT::UOUT:
        for (t = 0_UZ; t != this->seq_w; ++t) {
          this->Forward_Pass__Dropout__Uout__Loop(t, batch_size, n_inp,
                                                  layer->dropout_values[0],
                                                  unit_it->ptr_array_values);
        }
        break;
      default:
        break;
    }
  } else {
    switch (layer->type_dropout) {
      case LAYER_DROPOUT::BERNOULLI:
        for (t = 0_UZ; t != this->seq_w; ++t) {
          this->Forward_Pass__Dropout__Bernoulli__Inference__Loop(
              t, batch_size, n_inp, layer->dropout_values[0],
              unit_it->ptr_array_values);
        }
        break;
      default:
        break;
    }
  }
}

void Model::assign_inputs_pre_train_rec_mp(size_t const batch_size,
                                           real const *const *const Xm) {
  int const batch_size_(static_cast<int>(batch_size));
  int i;

  Layer *const layer(this->ptr_array_layers);

  AF_unit *const unit_it(layer->ptr_array_AF_units);

  size_t const n_inp(static_cast<size_t>(layer->ptr_last_AF_unit - unit_it));
  size_t t;

  // Loop through each sample data.
#pragma omp for schedule(static)
  for (i = 0; i < batch_size_; ++i)
    for (t = 0_UZ; t != this->seq_w; ++t)
      Mem::copy(Xm[i] + t * n_inp, Xm[i] + t * n_inp + n_inp,
                unit_it->ptr_array_values + static_cast<size_t>(i) * n_inp +
                    this->batch_size * n_inp * t);

  if (this->type_state_propagation == PROPAGATION::TRAINING &&
      this->pre_training_level == 1_UZ) {
    switch (layer->type_dropout) {
      case LAYER_DROPOUT::BERNOULLI:
        for (t = 0_UZ; t != this->seq_w; ++t) {
          this->Forward_Pass__Dropout__Bernoulli__Training__OpenMP(
              layer->ptr_array__mask__dropout__bernoulli, t, batch_size, n_inp,
              unit_it->ptr_array_values);
        }
        break;
      case LAYER_DROPOUT::BERNOULLI_INVERTED:
        for (t = 0_UZ; t != this->seq_w; ++t) {
          this->Forward_Pass__Dropout__Bernoulli__Inverted__OpenMP(
              layer->ptr_array__mask__dropout__bernoulli, t, batch_size, n_inp,
              layer->dropout_values[0] == 0_r ? 0_r
                                              : 1_r / layer->dropout_values[0],
              unit_it->ptr_array_values);
        }
        break;
      case LAYER_DROPOUT::GAUSSIAN:
        for (t = 0_UZ; t != this->seq_w; ++t) {
          this->Forward_Pass__Dropout__Gaussian__OpenMP(
              t, batch_size, n_inp, layer->dropout_values[0],
              unit_it->ptr_array_values);
        }
        break;
      case LAYER_DROPOUT::UOUT:
        for (t = 0_UZ; t != this->seq_w; ++t) {
          this->Forward_Pass__Dropout__Uout__OpenMP(t, batch_size, n_inp,
                                                    layer->dropout_values[0],
                                                    unit_it->ptr_array_values);
        }
        break;
      default:
        break;
    }
  } else {
    switch (layer->type_dropout) {
      case LAYER_DROPOUT::BERNOULLI:
        for (t = 0_UZ; t != this->seq_w; ++t) {
          this->Forward_Pass__Dropout__Bernoulli__Inference__OpenMP(
              t, batch_size, n_inp, layer->dropout_values[0],
              unit_it->ptr_array_values);
        }
        break;
      default:
        break;
    }
  }
}
}  // namespace DL