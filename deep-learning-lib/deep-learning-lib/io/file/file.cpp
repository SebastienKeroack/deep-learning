/* Copyright 2016, 2022 S�bastien K�roack. All Rights Reserved.

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
#include "deep-learning-lib/pch.hpp"

// File header:
#include "deep-learning-lib/io/file.hpp"

// Deep learning lib:
#include "deep-learning-lib/data/string.hpp"
#include "deep-learning-lib/device/system/info.hpp"
#include "deep-learning-lib/io/logger.hpp"

// Boost:
// `CRTDBG_NEW` is not compatible with `boost::spirit::x3`.
#ifdef _CRTDBG_MAP_ALLOC
#undef new
#endif

#include <boost/spirit/home/x3.hpp>

#ifdef _CRTDBG_MAP_ALLOC
#define new CRTDBG_NEW
#endif

// Standard:
#include <sys/stat.h>

#include <filesystem>
#include <iostream>

namespace DL::File {
bool create_directories(std::wstring const &path_name) {
  if (path_exist(path_name)) return true;
  return std::filesystem::create_directories(path_name);
}

bool create_directory(std::wstring const &path_name) {
  if (path_exist(path_name)) return true;
  return std::filesystem::create_directory(path_name);
}

bool create_file(std::wstring const &path_name) {
  if (path_exist(path_name)) return true;
  std::wofstream file(path_name, std::wofstream::out | std::wofstream::trunc);
  return file.good();
}

bool create_temp_file(std::wstring const &path_name) {
  std::wstring const path_name_ext(path_name + L".tmp");

  if (path_exist(path_name)) {
    if (path_exist(path_name_ext) &&
        std::filesystem::equivalent(path_name, path_name_ext))
      return true;

    return (std::filesystem::copy_file(
        path_name, path_name_ext,
        std::filesystem::copy_options::overwrite_existing));
  }

  return true;
}

bool delete_file(std::wstring const &path_name) {
  if (path_exist(path_name) == false) return true;
  return std::filesystem::remove(path_name);
}

bool delete_directory(std::wstring const &path_name) {
  if (path_exist(path_name) == false) return true;
  return std::filesystem::remove_all(path_name) != 0u;
}

bool delete_temp_file(std::wstring const &path_name) {
  std::wstring const path_name_ext(path_name + L".tmp");

  if (path_exist(path_name_ext)) return delete_file(path_name_ext);

  return true;
}

template <typename IFSTREAM>
bool iclose(IFSTREAM &file, std::wstring const &path_name) {
  if (file.fail()) {
    ERR(L"Logical error on i/o operation `%ls`.", path_name.c_str());
    return false;
  }

  file.close();

  return true;
}

template <typename IFSTREAM>
bool iopen(IFSTREAM &file, std::wstring const &path_name,
           std::ios_base::openmode const openmode) {
  if (path_exist(path_name) == false) {
    ERR(L"Could not find the following path `%ls`.", path_name.c_str());
    return false;
  } else if (recover_temp_file(path_name) == false) {
    ERR(L"An error has been triggered from the `recover_temp_file(%ls)` "
        L"function.",
        path_name.c_str());
    return false;
  }

  file.open(path_name, openmode);

  if (file.is_open() == false) {
    ERR(L"The file `%ls` could not be opened.", path_name.c_str());
    return false;
  } else if (file.eof()) {
    ERR(L"The file `%ls` is empty.", path_name.c_str());
    return false;
  }

  return true;
}

template <typename T>
bool parse_from_file(std::wifstream &file, std::wstring const &discard,
                     T &out) {
  std::wstring l;

  file >> l;

  if (file.fail()) {
    ERR(L"Can not read properly inside `%ls`.", l.c_str());
    return false;
  } else if (l.find(discard) == std::wstring::npos) {
    ERR(L"Can not find `%ls` in `%ls`.", discard.c_str(), l.c_str());
    return false;
  }

  file >> out;

  if (file.fail()) {
    ERR(L"Can not read properly from the file.");
    return false;
  }

  return true;
}

template <typename T>
bool parse_number(wchar_t *&char_it, wchar_t *const last_char, T &output) {
  auto assigner([&output](auto &ctx) { output = _attr(ctx); });

  auto initializer([&assigner]() {
    if constexpr (std::is_same<T, wchar_t>::value) {
      return boost::spirit::x3::char_[assigner];
    } else if constexpr (std::is_same<T, int>::value) {
      return boost::spirit::x3::int_[assigner];
    } else if constexpr (std::is_same<T, short>::value) {
      return boost::spirit::x3::short_[assigner];
    } else if constexpr (std::is_same<T, long>::value) {
      return boost::spirit::x3::long_[assigner];
    } else if constexpr (std::is_same<T, long long>::value) {
      return boost::spirit::x3::long_long[assigner];
    } else if constexpr (std::is_same<T, unsigned short>::value) {
      return boost::spirit::x3::ushort_[assigner];
    } else if constexpr (std::is_same<T, unsigned int>::value) {
      return boost::spirit::x3::uint_[assigner];
    } else if constexpr (std::is_same<T, unsigned long>::value) {
      return boost::spirit::x3::ulong_[assigner];
    } else if constexpr (std::is_same<T, unsigned long long>::value) {
      return boost::spirit::x3::ulong_long[assigner];
    } else if constexpr (std::is_same<T, float>::value) {
      return boost::spirit::x3::float_[assigner];
    } else if constexpr (std::is_same<T, double>::value) {
      return boost::spirit::x3::double_[assigner];
    } else {
      throw std::logic_error("NotImplementedException");
    }
  });

  return boost::spirit::x3::phrase_parse(char_it, last_char, initializer(),
                                         boost::spirit::x3::ascii::space);
}

bool path_exist(std::wstring const &path_name) {
  return std::filesystem::exists(path_name);
}

bool read_stream_block(size_t &block_size, size_t const desired_block_size,
                       size_t const step_block_size,
                       std::vector<wchar_t> &buffers, std::wifstream &file,
                       wchar_t const until_reach) {
  if (file.eof()) return true;

  // Current position.
  std::wstreampos const tellg(file.tellg());

  if (file.fail()) {
    ERR(L"Can not gets the current position from the input stream.");
    return false;
  }

  // Remaining characters based on current position.
  file.seekg(0, std::ios::end);

  if (file.fail()) {
    ERR(L"Can not sets the position of the input stream.");
    return false;
  }

  size_t const remaining_size(static_cast<size_t>(file.tellg() - tellg)),
      tmp_block_size(remaining_size < desired_block_size ? remaining_size
                                                         : desired_block_size);

  if (file.fail()) {
    ERR(L"Can not gets the position from the input stream.");
    return false;
  }

  // Return to the last position.
  file.seekg(tellg, std::ios::beg);

  if (file.fail()) {
    ERR(L"Can not sets the position of the input stream to the beginning.");
    return false;
  }

  // If not enough space in the buffer resize it.
  if (buffers.size() < tmp_block_size) buffers.resize(tmp_block_size);

  // Read block into buffers.
  file.read(&buffers[0], tmp_block_size);

  if (file.fail()) {
    ERR(L"Can not read properly the file.");
    return false;
  }

  // Store current block size.
  block_size = tmp_block_size;

  // If we continue to read until reach a specific character.
  if (until_reach != 0x00 && buffers[tmp_block_size - 1] != until_reach &&
      file.eof() == false && remaining_size != tmp_block_size) {
    // Do while until reach.
    do {
      // If not enough space in the buffer resize it.
      if (buffers.size() < block_size + 1)
        buffers.resize(block_size + 1 + step_block_size);

      // Read character into buffer.
      file.read(&buffers[block_size], 1);
    } while (file.eof() == false && file.fail() == false &&
             buffers[block_size++] != until_reach);

    if (file.fail()) {
      ERR(L"Can not read properly the character (%c).",
          buffers[block_size - 1]);
      return false;
    }
  }

  return true;
}

template <typename T>
bool read_stream_block_n_parse(wchar_t *&char_it, wchar_t *&last_char,
                               size_t &block_size,
                               size_t const desired_block_size,
                               size_t const step_block_size, T &output,
                               std::vector<wchar_t> &buffers,
                               std::wifstream &file,
                               wchar_t const until_reach) {
  if (char_it == last_char) {
    if (read_stream_block(block_size, desired_block_size, step_block_size,
                          buffers, file, until_reach) == false) {
      ERR(L"An error has been triggered from the `read_stream_block(%zu, %zu, "
          "%zu, vector, wifstream, %ls)` function.",
          block_size, desired_block_size, step_block_size, until_reach);
      return false;
    }

    char_it = &buffers[0];
    last_char = char_it + block_size;
  }

  if (parse_number<T>(char_it, last_char, output) == false) {
    ERR(L"An error has been triggered from the \"parse_number()\" function.");
    return false;
  }

  return true;
}

bool recover_temp_file(std::wstring const &path_name) {
  std::wstring const path_name_ext(path_name + L".tmp");

  if (path_exist(path_name_ext)) {
    if (std::filesystem::equivalent(path_name, path_name_ext) == false &&
        std::filesystem::copy_file(
            path_name_ext, path_name,
            std::filesystem::copy_options::overwrite_existing) == false) {
      ERR(L"An error has been triggered from the `copy_file(%ls, %ls)` "
          L"function.",
          path_name_ext.c_str(), path_name.c_str());

      return false;
    } else if (delete_file(path_name_ext) == false) {
      ERR(L"An error has been triggered from the `delete_file(%ls)` function.",
          path_name_ext.c_str());

      return false;
    }
  }

  return true;
}

bool wclose(std::wofstream &file, std::wstring const &path_name) {
  if (file.fail()) {
    ERR(L"Logical error on i/o operation `%ls`.", path_name.c_str());
    return false;
  }

  file.flush();

  if (file.fail()) {
    ERR(L"An error has been triggered from the `flush()` function. Logical "
        L"error on i/o operation `%ls`.",
        path_name.c_str());
    return false;
  }

  file.close();

  if (delete_temp_file(path_name) == false) {
    ERR(L"An error has been triggered from the `File::delete_temp_file(%ls)` "
        L"function.",
        path_name.c_str());
    return false;
  }

  return true;
}

bool wopen(std::wofstream &file, std::wstring const &path_name,
           std::ios_base::openmode const openmode) {
  if (create_temp_file(path_name) == false) {
    ERR(L"An error has been triggered from the `File::create_temp_file(%ls)` "
        L"function.",
        path_name.c_str());
    return false;
  }

  file.open(path_name, openmode);

  if (file.is_open() == false) {
    ERR(L"The file `%ls` could not be opened.", path_name.c_str());
    return false;
  }

  return true;
}

bool write(std::wstring const &path_name, std::wstring const &text,
           std::ios_base::openmode const openmode) {
  std::wofstream file(path_name, openmode);

  if (file.is_open() == false) return false;

  file.write(text.c_str(), text.size());

  file.flush();
  file.close();

  return true;
}
}  // namespace DL::File

// clang-format off
template bool DL::File::iopen<std::ifstream>(std::ifstream &, std::wstring const &, std::ios_base::openmode const);
template bool DL::File::iopen<std::wifstream>(std::wifstream &, std::wstring const &, std::ios_base::openmode const);
template bool DL::File::iclose<std::ifstream>(std::ifstream &, std::wstring const &);
template bool DL::File::iclose<std::wifstream>(std::wifstream &, std::wstring const &);
template bool DL::File::read_stream_block_n_parse<size_t>(wchar_t *&, wchar_t *&, size_t &, size_t const, size_t const, size_t &, std::vector<wchar_t> &, std::wifstream &, wchar_t const);
template bool DL::File::read_stream_block_n_parse<double>(wchar_t *&, wchar_t *&, size_t &, size_t const, size_t const, double &, std::vector<wchar_t> &, std::wifstream &, wchar_t const);
template bool DL::File::parse_from_file<std::wstring>(std::wifstream &, std::wstring const &, std::wstring &);
// clang-format on