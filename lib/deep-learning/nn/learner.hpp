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
#include "deep-learning/data/enum/hierarchy.hpp"
#include "deep-learning/data/time.hpp"
#include "deep-learning/drivers/driver.hpp"
#include "deep-learning/v1/learner/model.hpp"

// Eigen:
// `CRTDBG_NEW` is not compatible with `Eigen`.
#ifdef _CRTDBG_MAP_ALLOC
#undef new
#endif

#include <eigen3/Eigen/Dense>

#ifdef _CRTDBG_MAP_ALLOC
#define new CRTDBG_NEW
#endif

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

using Matrix3x2 =
    Eigen::Matrix<double, ENV::LENGTH, HIERARCHY::LENGTH, Eigen::RowMajor>;
using MapVec1x2 = Eigen::Map<Eigen::Vector<double, 2>>;

struct Cache {
  int last_trained_save_step = -(std::numeric_limits<int>::max)();
  int next_evalt_step = -(std::numeric_limits<int>::max)();
  int next_trainer_save_step = 0;
};

class Checkpoint {
 public:
  Checkpoint(std::wstring const &workdir, bool const load = false,
             std::wstring const &name = L"checkpoint");

  bool inited = false;
  bool is_top;
  bool load(void);
  bool operator()(void);
  bool save(void);

  int last_update_step = 0;

  void reset(void);
  void update(int const step);

  MapVec1x2 operator[](int const key);

  Matrix3x2 values;

  std::wstring path_name;
};

class Checkpointer {
 public:
  Checkpointer(std::wstring const &ckpt_dir, v1::Model *model,
               int const max_to_keep, int const &step_train);

  bool load(void);
  bool save(void);

  void rotate(void);

 private:
  bool save_model_checkpoint_path(void);

  int max_to_keep;
  int const &step_train;

  v1::Model *model;

  std::wstring ckpt_dir;
  std::wstring ckpt_file;
  std::wstring read_model_checkpoint_path(void);
};

struct Directories;

struct Checkpointers {
  Checkpointers(Directories *dirs, v1::Model *model, int const &step_train);
  ~Checkpointers(void);

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

  bool const log_g_fn(int const g) const;

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

template <typename T>
class Var {
 public:
  Var(std::wstring const &name, T const initial, std::wstring const &workdir,
      bool const load);

  bool load(void);
  bool save(void);

  T value;

 private:
  std::wstring path_name;
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
  void save(int const &g);
  void train(int const &g);
};
}  // namespace DL