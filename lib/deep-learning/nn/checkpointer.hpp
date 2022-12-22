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

#pragma once

// Deep learning:
#include "deep-learning/nn/checkpoint.hpp"
#include "deep-learning/v1/learner/model.hpp"

// Standard:
#include <string>

namespace DL {
class Checkpointer {
 protected:
  v1::Model *model;

 public:
  Checkpointer(std::wstring const &ckpt_dir, v1::Model *model,
               int const max_to_keep, int const &step_train);
  virtual ~Checkpointer(void);

  bool load(void);
  virtual bool operator()(int const &g, bool const force = false);
  bool save(void);

  void rotate(void);

 private:
  bool save_model_checkpoint_path(void);

  int max_to_keep;
  int const &step_train;

  std::wstring ckpt_dir;
  std::wstring ckpt_file;
  std::wstring read_model_checkpoint_path(void);
};

/* CheckpointerInterval:
  - Frequent disk usage. */
class CheckpointerInterval : public Checkpointer {
 public:
  CheckpointerInterval(std::wstring const &ckpt_dir, v1::Model *model,
                       int const max_to_keep, int const &step_train,
                       int const interval, Checkpoint *checkpoint = nullptr);

  bool operator()(int const &g, bool const force = false);

  int interval;
  int next_step = 0;

  Checkpoint *checkpoint;
};

/* CheckpointerTopDelayInterval:
  - High memory consumption. */
class CheckpointerTopDelayInterval : public Checkpointer {
 public:
  CheckpointerTopDelayInterval(std::wstring const &ckpt_dir, v1::Model *model,
                               int const max_to_keep, int const &step_train,
                               int const interval,
                               Checkpoint *checkpoint = nullptr);
  ~CheckpointerTopDelayInterval(void);

  bool copy_or_update(void);
  bool operator()(int const &g, bool const force = false);

  int interval;
  int next_step = std::numeric_limits<int>::max();

  Checkpoint *checkpoint;

  v1::Model *model2 = nullptr;
};
}  // namespace DL