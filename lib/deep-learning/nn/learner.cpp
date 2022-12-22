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
#include "deep-learning/data/enum/env.hpp"
#include "deep-learning/data/enum/hierarchy.hpp"
#include "deep-learning/data/string.hpp"
#include "deep-learning/io/file.hpp"
#include "deep-learning/io/logger.hpp"
#include "deep-learning/io/term/input.hpp"
#include "deep-learning/session.hpp"

// FMT:
#include <fmt/core.h>
#include <fmt/xchar.h>

using namespace DL::File;
using namespace DL::Str;
using namespace DL::Term;
using namespace DL::Time;

namespace DL {
constexpr wchar_t SEP_LINE[] =
    L"=================================================================";

Checkpointers::Checkpointers(Checkpoint *checkpoint, Directories *dirs,
                             v1::Model *model, int const &step_train,
                             int const interval)
    : trained(new CheckpointerTopDelayInterval(
          dirs->trained, model, 5, step_train, interval, checkpoint)),
      trainer(new CheckpointerInterval(dirs->trainer, model, 5, step_train,
                                       interval)) {}

Checkpointers::~Checkpointers(void) {
  delete (this->trained);
  delete (this->trainer);
}

bool Checkpointers::operator()(int const &g, bool const force) {
  bool saved;
  saved = (*this->trained)(g, force);
  saved = (*this->trainer)(g, force);
  return saved;
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

bool const Interval::log_g_fn(int const &g) const {
  if (this->log_g == 0) return false;
  return g % this->log_g == 0;
}

Timer::Timer(void) : tick(CHRONO_NOW()) {}

double Timer::operator()(void) {
  this->elapsed = chrono_cast(this->tick);
  this->tick += std::chrono::seconds(static_cast<long long>(this->elapsed));
  return this->elapsed;
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
      new Checkpointers(this->checkpoint, this->dirs, this->model,
                        this->step_globl->value, this->interval.save);

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
      this->cache.next_evalt_step = g + this->interval.evalt;
      this->evaluate(g);
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
  if (this->cache.next_evalt_step != g + this->interval.evalt - 1) {
    this->cache.next_evalt_step = g + this->interval.evalt;
    this->evaluate(g);
    this->save(g, true);
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

void Learner::save(int const &g, bool const force) {
  if (this->interval.save == 0) return;

  if ((*this->checkpointers)(g, force)) {
    this->checkpoint->save();
    this->step_globl->save();
  }
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