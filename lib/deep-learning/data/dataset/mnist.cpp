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
#include "deep-learning/data/dataset/mnist.hpp"

// Deep learning:
#include "deep-learning/data/string.hpp"
#include "deep-learning/io/file.hpp"
#include "deep-learning/io/logger.hpp"
#include "deep-learning/io/term/command.hpp"
#include "deep-learning/ops/math.hpp"

// FMT:
#include <fmt/core.h>
#include <fmt/xchar.h>

// Standard:
#include <iostream>

using namespace DL::File;
using namespace DL::Math;
using namespace DL::Str;

namespace DL {
enum FILE_IDX : int { IMAGE, LABEL };

MNIST::MNIST(std::wstring const &workdir) : Dataset(workdir){};

bool MNIST::download(std::wstring const &file_name,
                     std::wstring const &path_name) {
  std::wstring const uri(
      fmt::format(L"http://yann.lecun.com/exdb/mnist/{}.gz", file_name)
          .c_str());
  std::wstring const path_name_gz(fmt::format(L"{}.gz", path_name).c_str());

  return Term::download(uri.c_str(), path_name_gz.c_str()) &&
         Term::gunzip(path_name.c_str());
}

bool MNIST::load(ENV::TYPE const &env_type) {
  std::vector<std::wstring> files(2);

  if (ENV::TRAIN == env_type) {
    files[IMAGE] = L"train-images-idx3-ubyte";
    files[LABEL] = L"train-labels-idx1-ubyte";
  } else {
    files[IMAGE] = L"t10k-images-idx3-ubyte";
    files[LABEL] = L"t10k-labels-idx1-ubyte";
  }

  for (auto &file_name : files) {
    std::wstring const path_name(this->workdir + OS_SEP + file_name);
    if (path_exist(path_name) == false &&
        this->download(file_name, path_name) == false) {
      ERR(L"An error has been triggered from the `MNIST::download(%ls, %ls)` "
          L"function.",
          file_name.c_str(), path_name.c_str());
      return false;
    }
  }

  Shape const shape_inp(
      this->load_images(this->workdir + OS_SEP + files[IMAGE]));
  if (0 == shape_inp[0]) {
    ERR(L"The file `%ls` does not contain any images", files[IMAGE].c_str());
    return false;
  }

  Shape const shape_out(
      this->load_labels(this->workdir + OS_SEP + files[LABEL]));
  if (0 == shape_out[0]) {
    ERR(L"The file `%ls` does not contain any labels", files[LABEL].c_str());
    return false;
  }

  if (shape_inp[0] != shape_out[0]) {
    ERR(L"The number of images (%d) differs with the number of labels (%d).",
        shape_inp[0], shape_out[0]);
    return false;
  }

  this->_n_data = static_cast<size_t>(shape_inp[0]);
  this->_n_inp = static_cast<size_t>(shape_inp[1]);
  this->_n_out = static_cast<size_t>(shape_out[1]);

  return true;
}

void MNIST::print_sample(size_t const idx) const {
  real const *X(this->Xm[idx]);
  real const *Y(this->Ym[idx]);

  auto at([&](int row, int col) -> int {
    return static_cast<int>(round(X[row * this->_n_rows + col]));
  });

  std::wcout << fmt::format(L"X[{}]:", idx) << std::endl;
  for (int row(0), col(0); row != this->_n_rows; ++row) {
    for (col = 0; col != this->_n_cols - 1; ++col)
      std::wcout << at(row, col) << L" ";
    std::wcout << at(row, col);
    std::wcout << std::endl;
  }

  std::wcout << fmt::format(L"Y[{}]:", idx) << std::endl;
  size_t i;
  for (i = 0_UZ; i != this->n_out - 1_UZ; ++i) std::wcout << i << L" ";
  std::wcout << i << std::endl;
  for (i = 0_UZ; i != this->n_out - 1_UZ; ++i) std::wcout << Y[i] << L" ";
  std::wcout << Y[i] << std::endl << std::endl;
}

Shape const MNIST::load_images(std::wstring const &path_name) {
  Shape shape(std::vector<int>{0, 0});

  std::ifstream file;
  if (iopen(file, path_name, std::wios::in | std::wios::binary) == false) {
    ERR(L"An error has been triggered from the `File::iopen(%ls)` "
        L"function.",
        path_name.c_str());
    return shape;
  }

  int magic_num;
  file.read(reinterpret_cast<char *>(&magic_num), sizeof(int));
  if (2051 != (magic_num = reverse_int(magic_num))) {
    ERR(L"The file contains an invalid magic number. Saw `%d`.", magic_num);
    return shape;
  }

  // Read the number of images.
  file.read(reinterpret_cast<char *>(&shape[0]), sizeof(int));
  shape[0] = reverse_int(shape[0]);

#if DEEPLEARNING_USE_ADEPT
  WARN(L"Adept: reducing the number of samples from `%d` to `%d`", shape[0],
       static_cast<int>(shape[0] * 0.01));
  shape[0] = static_cast<int>(shape[0] * 0.01);
#endif

  file.read(reinterpret_cast<char *>(&this->_n_rows), sizeof(int));
  this->_n_rows = reverse_int(this->_n_rows);

  file.read(reinterpret_cast<char *>(&this->_n_cols), sizeof(int));
  this->_n_cols = reverse_int(this->_n_cols);

  shape[1] = this->_n_rows * this->_n_cols;
  this->Xm = new real const *[shape[0]];
  this->X = new real[shape.num_elements()];

  unsigned char val;
  for (int i(0); i != shape[0]; ++i) {
    this->Xm[i] = this->X + i * shape[1];

    for (int pixel(0); pixel != shape[1]; ++pixel) {
      file.read(reinterpret_cast<char *>(&val), sizeof(unsigned char));
      this->X[i * shape[1] + pixel] = static_cast<real>(val) / 255_r;
    }
  }

  if (iclose(file, path_name) == false)
    ERR(L"An error has been triggered from the `File::iclose(%ls)` "
        L"function.",
        path_name.c_str());
  return shape;
}

Shape const MNIST::load_labels(std::wstring const &path_name) {
  Shape shape(std::vector<int>{0, 10});

  std::ifstream file;
  if (iopen(file, path_name, std::wios::in | std::wios::binary) == false) {
    ERR(L"An error has been triggered from the `File::iopen(%ls)` "
        L"function.",
        path_name.c_str());
    return shape;
  }

  int magic_num;
  file.read(reinterpret_cast<char *>(&magic_num), sizeof(int));
  if (2049 != (magic_num = reverse_int(magic_num))) {
    ERR(L"The file contains an invalid magic number. Saw `%d`.", magic_num);
    return shape;
  }

  // Read the number of labels.
  file.read(reinterpret_cast<char *>(&shape[0]), sizeof(int));
  shape[0] = reverse_int(shape[0]);

#if DEEPLEARNING_USE_ADEPT
  shape[0] = static_cast<int>(shape[0] * 0.01);
#endif

  this->Ym = new real const *[shape[0]];
  this->Y = new real[shape.num_elements()]();

  unsigned char key;
  for (int i(0); i != shape[0]; ++i) {
    this->Ym[i] = this->Y + i * shape[1];

    file.read(reinterpret_cast<char *>(&key), sizeof(unsigned char));

    this->Y[i * shape[1] + key] = 1_r;
  }

  if (iclose(file, path_name) == false)
    ERR(L"An error has been triggered from the `File::iclose(%ls)` "
        L"function.",
        path_name.c_str());
  return shape;
}
}  // namespace DL