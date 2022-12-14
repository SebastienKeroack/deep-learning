/* Copyright 2016, 2022 Sébastien Kéroack. All Rights Reserved.

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

// Standard:
#include <fstream>
#include <string>
#include <vector>

namespace DL::File {
bool create_directories(std::wstring const &path_name);

bool create_directory(std::wstring const &path_name);

bool create_file(std::wstring const &path_name);

bool create_temp_file(std::wstring const &path_name);

bool delete_file(std::wstring const &path_name);

bool delete_directory(std::wstring const &path_name);

bool delete_temp_file(std::wstring const &path_name);

template <typename IFSTREAM>
bool iclose(IFSTREAM &file, std::wstring const &path_name);

template <typename IFSTREAM>
bool iopen(IFSTREAM &file, std::wstring const &path_name,
           std::ios_base::openmode const openmode);

template <typename T>
bool parse_from_file(std::wifstream &file, std::wstring const &discard, T &out);

bool path_exist(std::wstring const &path_name);

template <typename T>
bool read_stream_block_n_parse(
    wchar_t *&char_it, wchar_t *&last_char, size_t &block_size,
    size_t const desired_block_size, size_t const step_block_size,
    T &ref_output_received, std::vector<wchar_t> &buffers, std::wifstream &file,
    wchar_t const until_reach = 0x00);

bool recover_temp_file(std::wstring const &path_name);

bool wclose(std::wofstream &file, std::wstring const &path_name);

bool wopen(std::wofstream &file, std::wstring const &path_name,
           std::ios_base::openmode const openmode);

bool write(std::wstring const &path_name, std::wstring const &text,
           std::ios_base::openmode const openmode);

std::wstring home_directory(void);
}  // namespace DL::File