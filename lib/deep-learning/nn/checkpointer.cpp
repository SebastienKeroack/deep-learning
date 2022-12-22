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
#include "deep-learning/nn/checkpointer.hpp"

// Deep learning:
#include "deep-learning/data/enum/hierarchy.hpp"
#include "deep-learning/data/string.hpp"
#include "deep-learning/io/file.hpp"
#include "deep-learning/io/flags.hpp"
#include "deep-learning/io/logger.hpp"

// FMT:
#include <fmt/core.h>
#include <fmt/xchar.h>

// Standard:
#include <filesystem>
#include <tuple>

using namespace DL::File;
using namespace DL::Str;

namespace DL {
Checkpointer::Checkpointer(std::wstring const &ckpt_dir, v1::Model *model,
                           int const max_to_keep, int const &step_train)
    : model(model),
      max_to_keep(max_to_keep),
      step_train(step_train),
      ckpt_dir(ckpt_dir),
      ckpt_file(ckpt_dir + OS_SEP + L"checkpoint") {}

Checkpointer::~Checkpointer(void) {}

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

bool Checkpointer::operator()(int const &g, bool const force) {
  throw std::logic_error("NotImplementedException");
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

CheckpointerInterval::CheckpointerInterval(
    std::wstring const &ckpt_dir, v1::Model *model, int const max_to_keep,
    int const &step_train, int const interval, Checkpoint *checkpoint)
    : Checkpointer(ckpt_dir, model, max_to_keep, step_train),
      interval(interval),
      checkpoint(checkpoint) {}

bool CheckpointerInterval::operator()(int const &g, bool const force) {
  if (!flags->save) return false;

  bool const nullptr_or_is_top(!this->checkpoint || this->checkpoint->is_top);
  bool saved(false);

  if (force || (g >= this->next_step && nullptr_or_is_top)) {
    this->next_step = g + this->interval;

    try {
      this->save();
    } catch (std::exception const &err) {
      ERR(L"Caught exception: %s", to_wstring(err.what()).c_str());
    }

    saved = true;
  }

  return saved;
}

CheckpointerTopDelayInterval::CheckpointerTopDelayInterval(
    std::wstring const &ckpt_dir, v1::Model *model, int const max_to_keep,
    int const &step_train, int const interval, Checkpoint *checkpoint)
    : Checkpointer(ckpt_dir, model, max_to_keep, step_train),
      interval(interval),
      checkpoint(checkpoint) {}

CheckpointerTopDelayInterval::~CheckpointerTopDelayInterval(void) {
  delete (this->model2);
}

bool CheckpointerTopDelayInterval::copy_or_update(void) {
  if (nullptr == this->model2) {
    this->model2 = new v1::Model;

    if (!this->model2->copy(*this->model, false, false)) {
      ERR(L"An error has been triggered from the "
          L"`Model::copy()` function.");
      return false;
    }
  } else if (!this->model2->update(*this->model, false, false)) {
    ERR(L"An error has been triggered from the "
        L"`Model::update()` function.");
    return false;
  }

  return true;
}

bool CheckpointerTopDelayInterval::operator()(int const &g, bool const force) {
  if (!flags->save) return false;

  bool saved(false);

  if (this->checkpoint->is_top && g != this->next_step) {
    this->next_step = g + this->interval;
    this->copy_or_update();
  }

  if ((force || g >= this->next_step) && this->model2) {
    this->next_step = std::numeric_limits<int>::max();

    v1::Model *original(this->model);

    try {
      this->model = this->model2;
      this->save();
    } catch (std::exception const &err) {
      ERR(L"Caught exception: %s", to_wstring(err.what()).c_str());
    }

    this->model = original;
    saved = true;
  }

  return saved;
}
}  // namespace DL