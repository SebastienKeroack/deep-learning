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

#include "deep-learning-lib/pch.hpp"

#include "deep-learning-lib/v1/learner/model.hpp"

namespace DL::v1 {
void Model::compute_r(size_t const batch_size, real const *const *const Ym) {
  switch (this->type) {
    case MODEL::AUTOENCODER:
      if (this->pre_training_level != 0_UZ && this->_initialized__weight) {
        this->compute_r_pre_train(batch_size);
        break;
      }
    default:
      if (this->seq_w > 1_UZ) {
        if (this->use_mp && this->is_mp_initialized)
          this->compute_r_rec_mp(batch_size, Ym);
        else
          this->compute_r_rec_st(batch_size, Ym);
      } else {
        if (this->use_mp && this->is_mp_initialized)
          this->compute_r_fwp_mp(batch_size, Ym);
        else
          this->compute_r_fwp_st(batch_size, Ym);
      }
      break;
  }
}

void Model::compute_r_pre_train(size_t const batch_size) {
  if (this->seq_w > 1_UZ) {
    if (this->use_mp && this->is_mp_initialized)
      this->compute_r_pre_train_rec_mp(batch_size);
    else
      this->compute_r_pre_train_rec_st(batch_size);
  } else {
    if (this->use_mp && this->is_mp_initialized)
      this->compute_r_pre_train_fwp_mp(batch_size);
    else
      this->compute_r_pre_train_fwp_st(batch_size);
  }
}
}
