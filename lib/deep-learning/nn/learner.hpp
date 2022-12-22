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
#include "deep-learning/data/time.hpp"
#include "deep-learning/drivers/driver.hpp"
#include "deep-learning/nn/checkpoint.hpp"
#include "deep-learning/nn/checkpointer.hpp"
#include "deep-learning/nn/var.hpp"
#include "deep-learning/v1/learner/model.hpp"

// Standard:
#include <memory>

namespace DL {
// clang-format off
struct DCLEAN {
  typedef enum : int {
    UNUSED,
    SOFT,
    PURGE
  } TYPE;
};
// clang-format on

struct Cache {
  int next_evalt_step = -(std::numeric_limits<int>::max)();
};

struct Directories;

struct Checkpointers {
  Checkpointers(Checkpoint *checkpoint, Directories *dirs, v1::Model *model,
                int const &step_train, int const interval);
  ~Checkpointers(void);

  bool operator()(int const &g, bool const force = false);

  Checkpointer *trained;
  Checkpointer *trainer;
};

struct Directories {
  Directories(std::wstring const &workdir, std::wstring const &name);

  bool initialize(void);

  std::wstring workdir;
  std::wstring datasets;
  std::wstring trained;
  std::wstring trainer;
};

class Interval {
 public:
  Interval(float early_stop_pct = 1.0, int n_iters = 1, int evalt = 0,
           int log_g = 0, int save = 0);

  bool const log_g_fn(int const &g) const;

  int early_stop = 1;
  int evalt = 0;
  int log_g = 0;
  int save = 0;
};

struct Timer {
  Timer(void);

  double elapsed = 0.0;
  double operator()(void);

  TIME_POINT tick;
};

class Learner {
 protected:
  virtual void _create_datasets(void) = 0;
  virtual void _create_model(void) = 0;

 public:
  Learner(std::wstring const &name, int n_iters);
  ~Learner(void);

  int n_iters;

  void initialize(void);
  void optimize(void);

  Cache cache;

  Checkpoint *checkpoint = nullptr;

  Checkpointers *checkpointers = nullptr;

  Directories *dirs = nullptr;

  DCLEAN::TYPE _clean;

  Interval interval;

  v1::Model *model = nullptr;

  Timer timer;

  Var<int> *step_globl = nullptr;

  std::vector<std::unique_ptr<Dataset>> datasets;
  std::vector<Driver> drivers;

  std::wstring name;

 private:
  void _create_checkpoint(void);
  void _create_checkpointers(void);
  void _create_drivers(void);
  void _create_step_globl(void);
  void _initialize_dirs(void);
  void _optimize(void);
  void evaluate(int const &g);
  void log_metrics(wchar_t const *const title);
  void save(int const &g, bool const force = false);
  void train(int const &g);
};
}  // namespace DL