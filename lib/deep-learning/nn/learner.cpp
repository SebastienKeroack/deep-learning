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

// PCH:
#include "pch.hpp"

// File header:
#include "deep-learning/nn/learner.hpp"

// Deep learning:
#include "deep-learning/data/enum/hierarchy.hpp"
#include "deep-learning/data/string.hpp"
#include "deep-learning/data/time.hpp"
#include "deep-learning/drivers/driver.hpp"
#include "deep-learning/io/file.hpp"
#include "deep-learning/io/term/input.hpp"
#include "deep-learning/session.hpp"

// FMT:
#include <fmt/core.h>
#include <fmt/xchar.h>

// Standard:
#include <filesystem>
#include <tuple>

using namespace DL::File;
using namespace DL::Str;
using namespace DL::Term;
using namespace DL::Time;

namespace DL {
constexpr wchar_t SEP_LINE[] =
    L"=================================================================";

Checkpoint::Checkpoint(std::wstring const &workdir, bool const load,
                       std::wstring const &name) {
  this->path_name = workdir + OS_SEP + name;

  if (load) this->load();
  if (this->inited == false) this->reset();
}

bool Checkpoint::load(void) {
  if (path_exist(this->path_name) == false) {
    DEBUG(L"No such file `%ls`.", this->path_name.c_str());
    return false;
  }

  std::wifstream file;
  if (iopen(file, this->path_name, std::wios::in | std::wios::binary) ==
      false) {
    ERR(L"An error has been triggered from the "
        L"`File::iopen(%ls)` function.",
        this->path_name.c_str());
    return false;
  }

  double *data(this->values.data());
  for (int i(0); i != this->values.size(); ++i) file >> data[i] >> std::ws;

  if (iclose(file, this->path_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`File::iclose(%ls)` function.",
        this->path_name.c_str());
    return false;
  }

  this->operator()();

  return true;
}

bool Checkpoint::operator()(void) {
  MapVec1x2 const train(this->values.data() + ENV::TRAIN * 2);
  MapVec1x2 const valid(this->values.data() + ENV::VALID * 2);

  this->is_top = (((train[HIERARCHY::TRAINER] <= train[HIERARCHY::TRAINED] ||
                    train[HIERARCHY::TRAINER] <= valid[HIERARCHY::TRAINED]) &&
                   valid[HIERARCHY::TRAINER] < valid[HIERARCHY::TRAINED]) ||
                  ((valid[HIERARCHY::TRAINER] <= valid[HIERARCHY::TRAINED] ||
                    valid[HIERARCHY::TRAINER] <= train[HIERARCHY::TRAINED]) &&
                   train[HIERARCHY::TRAINER] < train[HIERARCHY::TRAINED])) ||
                 this->inited == false;

  this->inited = true;

  return this->is_top;
}

bool Checkpoint::save(void) {
  std::wofstream file;
  if (wopen(file, this->path_name, std::wios::out | std::wios::binary) ==
      false) {
    ERR(L"An error has been triggered from the "
        L"`File::wopen(%ls)` function.",
        this->path_name.c_str());
    return false;
  }

  file << std::setprecision(15);
  double const *data(this->values.data());
  for (int i(0); i != this->values.size(); ++i) file << data[i] << L" ";

  if (wclose(file, this->path_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`File::wclose(%ls)` function.",
        this->path_name.c_str());
    return false;
  }

  return true;
}

void Checkpoint::reset(void) {
  this->values.setZero();
  this->inited = false;
  this->is_top = false;
}

void Checkpoint::update(int const step = 0) {
  this->values.col(HIERARCHY::TRAINED) = this->values.col(HIERARCHY::TRAINER);
  this->last_update_step = step;
}

MapVec1x2 Checkpoint::operator[](int const key) {
  return MapVec1x2(this->values.data() + key * 2);
}

Checkpointer::Checkpointer(std::wstring const &ckpt_dir, v1::Model *model,
                           int const max_to_keep, int const &step_train)
    : max_to_keep(max_to_keep),
      step_train(step_train),
      model(model),
      ckpt_dir(ckpt_dir),
      ckpt_file(ckpt_dir + OS_SEP + L"checkpoint") {}

bool Checkpointer::load(void) {
  std::wstring const &ckpt_name(this->read_model_checkpoint_path());
  if (ckpt_name.empty()) return false;

  std::wstring const path_weights(this->ckpt_dir + OS_SEP + ckpt_name + OS_SEP +
                                  L"model.net");
  if (path_exist(path_weights) == false) {
    DEBUG(L"No such file `%ls`.", path_weights.c_str());
    return false;
  }

  std::wstring const path_configs(this->ckpt_dir + OS_SEP + ckpt_name + OS_SEP +
                                  L"model.nn");
  if (path_exist(path_configs) == false) {
    DEBUG(L"No such file `%ls`.", path_configs.c_str());
    return false;
  }

  size_t const allowable_host_mem(this->model->maximum_allowable_memory_bytes);
  size_t allowable_devc_mem(allowable_host_mem);

  if (this->model->load(path_weights, path_configs, allowable_host_mem) ==
      false) {
    ERR(L"An error has been triggered from the "
        L"`Model::load()` function.");
    return false;
  }

  if (this->model->set_mp(this->model->use_mp) == false) {
    ERR(L"An error has been triggered from the "
        L"`Model::set_mp()` function.");
    return false;
  }

  if (this->model->set_cu(this->model->use_cu, allowable_devc_mem) == false) {
    ERR(L"An error has been triggered from the "
        L"`Model::set_cu()` function.");
    return false;
  }

  return true;
}

bool Checkpointer::save(void) {
  std::wstring const path_ckpt(this->ckpt_dir + OS_SEP + L"ckpt-" +
                               fmt::to_wstring(this->step_train));
  if (create_directory(path_ckpt) == false) {
    ERR(L"An error has been triggered from the "
        L"`File::create_directory(%ls)` function.",
        path_ckpt.c_str());
    return false;
  }

  std::wstring const path_weights(path_ckpt + OS_SEP + L"model.net");
  if (this->model->save_params(path_weights) == false) {
    ERR(L"An error has been triggered from the "
        L"`Model::save_params(%ls)` function.",
        path_weights.c_str());
    return false;
  }

  std::wstring const path_configs(path_ckpt + OS_SEP + L"model.nn");
  if (this->model->save_spec_params(path_configs) == false) {
    ERR(L"An error has been triggered from the "
        L"`Model::save_spec_params(%ls)` function.",
        path_configs.c_str());
    return false;
  }

  if (this->save_model_checkpoint_path() == false) {
    ERR(L"An error has been triggered from the "
        L"`save_model_checkpoint_path()` "
        L"function.");
    return false;
  }

  this->rotate();

  return true;
}

void Checkpointer::rotate(void) {
  std::vector<std::tuple<std::wstring, std::chrono::system_clock::duration>>
      ckpts;

  for (auto const &entry :
       std::filesystem::directory_iterator(CP_STR(this->ckpt_dir))) {
    if (entry.is_directory() == false) continue;
    if (entry.path().compare(L__("ckpt-")) == 0) continue;

    ckpts.push_back(
        std::make_tuple(entry.path().generic_wstring(),
                        entry.last_write_time().time_since_epoch()));
  }

  int const total_checkpoints(static_cast<int>(ckpts.size()));
  if (total_checkpoints <= this->max_to_keep) return;

  std::sort(ckpts.begin(), ckpts.end(), [](auto const &t1, auto const &t2) {
    return std::get<1>(t1) > std::get<1>(t2);
  });

  for (int i(this->max_to_keep); i != total_checkpoints; ++i)
    delete_directory(std::get<0>(ckpts[i]));
}

bool Checkpointer::save_model_checkpoint_path(void) {
  std::wofstream file;
  if (wopen(file, this->ckpt_file, std::wios::out | std::wios::binary) ==
      false) {
    ERR(L"An error has been triggered from the "
        L"`File::wopen(%ls)` function.",
        this->ckpt_file.c_str());
    return false;
  }

  file << L"model_checkpoint_path: ckpt-" << this->step_train;

  if (wclose(file, this->ckpt_file) == false) {
    ERR(L"An error has been triggered from the "
        L"`File::wclose(%ls)` function.",
        this->ckpt_file.c_str());
    return false;
  }

  return true;
}

std::wstring Checkpointer::read_model_checkpoint_path(void) {
  if (path_exist(this->ckpt_file) == false) {
    DEBUG(L"No such file `%ls`.", this->ckpt_file.c_str());
    return L"";
  }

  std::wifstream file;
  if (iopen(file, this->ckpt_file, std::wios::in | std::wios::binary) ==
      false) {
    ERR(L"An error has been triggered from the "
        L"`File::iopen(%ls)` function.",
        this->ckpt_file.c_str());
    return L"";
  }

  std::wstring ckpt_name;
  if (parse_from_file(file, L"model_checkpoint_path:", ckpt_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`File::parse_from_file()` function.",
        this->ckpt_file.c_str());
    return L"";
  }

  if (iclose(file, this->ckpt_file) == false) {
    ERR(L"An error has been triggered from the "
        L"`File::iclose(%ls)` function.",
        this->ckpt_file.c_str());
    return L"";
  }

  return ckpt_name;
}

Checkpointers::Checkpointers(Directories *dirs, v1::Model *model,
                             int const &step_train)
    : trained(new Checkpointer(dirs->trained, model, 5, step_train)),
      trainer(new Checkpointer(dirs->trainer, model, 5, step_train)) {}

Checkpointers::~Checkpointers(void) {
  delete (this->trained);
  delete (this->trainer);
}

Directories::Directories(std::wstring const &workdir,
                         std::wstring const &name) {
  this->workdir = workdir + OS_SEP + L"models" + OS_SEP + name;
  this->datasets = workdir + OS_SEP + L"datasets";
  this->trained = this->workdir + OS_SEP + L"trained";
  this->trainer = this->workdir + OS_SEP + L"trainer";
}

bool Directories::initialize(void) {
  if (create_directories(this->workdir) == false) {
    ERR(L"An error has been triggered from the "
        L"`File::create_directories(%ls)` function.",
        this->workdir.c_str());
    return false;
  }

  if (create_directory(this->datasets) == false) {
    ERR(L"An error has been triggered from the "
        L"`File::create_directory(%ls)` function.",
        this->datasets.c_str());
    return false;
  }

  if (create_directory(this->trained) == false) {
    ERR(L"An error has been triggered from the "
        L"`File::create_directory(%ls)` function.",
        this->trained.c_str());
    return false;
  }

  if (create_directory(this->trainer) == false) {
    ERR(L"An error has been triggered from the "
        L"`File::create_directory(%ls)` function.",
        this->trainer.c_str());
    return false;
  }

  return true;
}

Interval::Interval(float early_stop_pct, int n_iters, int evalt, int log_g,
                   int save) {
  this->early_stop = std::max<int>(
      1, static_cast<int>(static_cast<float>(n_iters) * early_stop_pct));
  this->evalt = evalt;
  this->log_g = log_g;
  this->save = save ? save : static_cast<int>(flags->save) * 25;
}

bool const Interval::log_g_fn(int const g) const {
  if (this->log_g == 0) return false;
  return g % this->log_g == 0;
}

Timer::Timer(void) : tick(CHRONO_NOW()) {}

double Timer::operator()(void) {
  this->elapsed = chrono_cast(this->tick);
  this->tick += std::chrono::seconds(static_cast<long long>(this->elapsed));
  return this->elapsed;
}

template <typename T>
Var<T>::Var(std::wstring const &name, T const initial,
            std::wstring const &workdir, bool const load)
    : value(initial) {
  this->path_name = workdir + OS_SEP + name;
  if (load) this->load();
}

template <typename T>
bool Var<T>::save(void) {
  std::wofstream file;
  if (wopen(file, this->path_name, std::wios::out | std::wios::binary) ==
      false) {
    ERR(L"An error has been triggered from the "
        L"`File::wopen(%ls)` function.",
        this->path_name.c_str());
    return false;
  }

  file << this->value;

  if (wclose(file, this->path_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`File::wclose(%ls)` function.",
        this->path_name.c_str());
    return false;
  }

  return true;
}

template <typename T>
bool Var<T>::load(void) {
  if (path_exist(this->path_name) == false) {
    DEBUG(L"No such file `%ls`.", this->path_name.c_str());
    return false;
  }

  std::wifstream file;
  if (iopen(file, this->path_name, std::wios::in | std::wios::binary) ==
      false) {
    ERR(L"An error has been triggered from the "
        L"`File::iopen(%ls)` function.",
        this->path_name.c_str());
    return false;
  }

  file >> this->value;

  if (iclose(file, this->path_name) == false) {
    ERR(L"An error has been triggered from the "
        L"`File::iclose(%ls)` function.",
        this->path_name.c_str());
    return false;
  }

  return true;
}

Learner::Learner(std::wstring const &name, int n_iters)
    : _clean(flags->load ? DCLEAN::UNUSED : DCLEAN::SOFT), name(name) {
  if (0 == n_iters) n_iters = parse_discrete(1, L"n_iters: ");
  this->n_iters = n_iters;

  this->interval = Interval(1.0f, n_iters);
}

Learner::~Learner(void) {
  delete (this->checkpoint);
  delete (this->checkpointers);
  delete (this->dirs);
  delete (this->model);
  delete (this->step_globl);
}

void Learner::initialize(void) {
  INFO(L"Initialize directories");
  this->_initialize_dirs();

  INFO(L"Create step global...");
  this->_create_step_globl();

  INFO(L"Create datasets...");
  this->_create_datasets();

  INFO(L"Create checkpoint...");
  this->_create_checkpoint();

  INFO(L"Create model...");
  this->_create_model();

  INFO(L"Create checkpointers...");
  this->_create_checkpointers();

  INFO(L"Create drivers...");
  this->_create_drivers();
}

void Learner::optimize(void) {
  TIME_POINT const t0(CHRONO_NOW());
  this->log_metrics(L"Begin optimization");

  this->_optimize();

  this->log_metrics(fmt::format(L"Optimization has endded after {}",
                                time_format(chrono_cast(t0)))
                        .c_str());
}

void Learner::_create_checkpoint(void) {
  this->checkpoint = new Checkpoint(this->dirs->workdir, flags->load);
}

void Learner::_create_checkpointers(void) {
  if ((flags->load | static_cast<bool>(this->interval.save)) == false) return;

  if (flags->load && DCLEAN::UNUSED == this->_clean)
    INFO(L"Restore checkpoints...");
  else
    INFO(L"Initialize checkpoints...");

  this->checkpointers =
      new Checkpointers(this->dirs, this->model, this->step_globl->value);

  if (flags->load) this->checkpointers->trainer->load();
}

void Learner::_create_drivers(void) {
  int const n_envs(static_cast<int>(this->datasets.size()));
  for (int i(0); i != n_envs; ++i)
    this->drivers.push_back(
        Driver(*this->datasets[ENV::TYPE(i)], ENV::TYPE(i), *this->model));
}

void Learner::_create_step_globl(void) {
  this->step_globl =
      new Var<int>(L"step_globl", 0, this->dirs->workdir, flags->load);
}

void Learner::_initialize_dirs(void) {
  this->dirs = new Directories(session->workdir, this->name);

  switch (this->_clean) {
    case DCLEAN::PURGE:
      delete_directory(this->dirs->workdir);
      break;
    case DCLEAN::SOFT:
      delete_file(this->dirs->workdir + OS_SEP + L"checkpoint");
      delete_file(this->dirs->workdir + OS_SEP + L"step_globl");
      delete_directory(this->dirs->trained);
      delete_directory(this->dirs->trainer);
      break;
    default:
      break;
  }

  if (this->dirs->initialize() == false)
    throw std::runtime_error(
        "An error has been triggered from the "
        "`Directories::initialize()` function.");
}

void Learner::_optimize(void) {
  bool early_stop(false);

  int const g_range_stop(this->step_globl->value + this->n_iters);
  int g_early_stop(1 + this->step_globl->value + this->interval.early_stop);
  int g(this->step_globl->value);

  for (; g != g_range_stop; ++g) {
    this->step_globl->value += 1;

    this->train(g);

    if (g >= this->cache.next_evalt_step) {
      this->evaluate(g);
      this->cache.next_evalt_step = g + this->interval.evalt;
    }

    this->save(g);

    if (this->checkpoint->is_top)
      g_early_stop = 1 + g + this->interval.early_stop;
    else
      early_stop = g == g_early_stop;

    if (this->interval.log_g_fn(g)) {
      if (this->checkpoint->is_top == false && g + 1 < g_range_stop &&
          early_stop == false)
        this->log_metrics(L"Stats");
      INFO(L"[%d]: Time elapsed: %ls", g, time_format(this->timer()).c_str());
    }

    if (early_stop) {
      WARN(L"Early stopping!");
      break;
    }
  }

  // Last evaluation.
  if (this->cache.next_evalt_step != g + this->interval.evalt) {
    this->cache.next_evalt_step = g + this->interval.evalt;
    this->cache.next_trainer_save_step = g;
    this->evaluate(g);
    this->save(g);
  }
}

void Learner::evaluate(int const &g) {
  TIME_POINT const t0(CHRONO_NOW());
  INFO(L"[%d]: Evaluating...", g);

  this->drivers[ENV::VALID].evalt();

  INFO(L"[%d]: Loss %f | Acc %f", g, this->model->get_loss(ENV::VALID),
       this->model->get_accu(ENV::VALID));
  INFO(L"[%d]:\tTime elapsed: %ls", g, time_format(chrono_cast(t0)).c_str());

  int const n_envs(static_cast<int>(this->datasets.size()));
  for (int i(0); i != n_envs; ++i)
    (*this->checkpoint)[ENV::TYPE(i)](HIERARCHY::TRAINER) =
        this->model->get_loss(ENV::TYPE(i));

  if ((*this->checkpoint)()) {
    Matrix3x2 const &values(this->checkpoint->values);

    // clang-format off
      INFO(
          L"Checkpoint!\n%ls\n"
          L"Trained:\n"
          L"  Train set:\n"
          L"    %.6f -> %.6f, %+.6f\n"
          L"  Valid set:\n"
          L"    %.6f -> %.6f, %+.6f\n"
          L"%ls",
          SEP_LINE,
          values(ENV::TRAIN, HIERARCHY::TRAINED), values(ENV::TRAIN, HIERARCHY::TRAINER),
          values(ENV::TRAIN, HIERARCHY::TRAINER) - values(ENV::TRAIN, HIERARCHY::TRAINED),
          values(ENV::VALID, HIERARCHY::TRAINED), values(ENV::VALID, HIERARCHY::TRAINER),
          values(ENV::VALID, HIERARCHY::TRAINER) - values(ENV::VALID, HIERARCHY::TRAINED),
          SEP_LINE);
    // clang-format on

    this->checkpoint->update(g);
  }
}

void Learner::log_metrics(wchar_t const *const title) {
  Matrix3x2 const &values(this->checkpoint->values);

  // clang-format off
    INFO(
      L"%ls\n%ls\n" 
      L"Trainer | Trained:\n"
      L"  Train set:\n"
      L"    %.6f | %.6f\n"
      L"  Valid set:\n"
      L"    %.6f | %.6f\n"
      L"  Last update step: %d\n"
      L"%ls",
      title, SEP_LINE,
      values(ENV::TRAIN, HIERARCHY::TRAINER), values(ENV::TRAIN, HIERARCHY::TRAINED),
      values(ENV::VALID, HIERARCHY::TRAINER), values(ENV::VALID, HIERARCHY::TRAINED),
      this->checkpoint->last_update_step,
      SEP_LINE);
  // clang-format on
}

void Learner::save(int const &g) {
  if (this->interval.save == 0) return;

  bool has_saved(false);

  if (this->checkpoint->is_top && g != this->cache.last_trained_save_step) {
    this->cache.last_trained_save_step = g;

    this->checkpoint->save();

    if (flags->save) {
      try {
        this->checkpointers->trained->save();
      } catch (std::exception const &err) {
        ERR(L"Caught exception: %s", to_wstring(err.what()).c_str());
      }
      has_saved = true;
    }
  }

  if (g >= this->cache.next_trainer_save_step) {
    this->cache.next_trainer_save_step = g + this->interval.save;

    if (flags->save) {
      try {
        this->checkpointers->trainer->save();
      } catch (std::exception const &err) {
        ERR(L"Caught exception: %s", to_wstring(err.what()).c_str());
      }
      has_saved = true;
    }
  }

  if (has_saved) this->step_globl->save();
}

void Learner::train(int const &g) {
  TIME_POINT const t0(CHRONO_NOW());
  INFO(L"[%d]: Training...", g);

  this->drivers[ENV::TRAIN].train();

  INFO(L"[%d]: Loss %f | Acc %f", g, this->model->get_loss(ENV::TRAIN),
       this->model->get_accu(ENV::TRAIN));
  INFO(L"[%d]:\tTime elapsed: %ls", g, time_format(chrono_cast(t0)).c_str());
}
}  // namespace DL

// clang-format off
template class DL::Var<int>;
// clang-format on