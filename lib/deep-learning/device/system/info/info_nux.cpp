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

// PCH:
#include "pch.hpp"

// File header:
#include "deep-learning/device/system/info.hpp"

// Deep learning:
#include "deep-learning/data/string.hpp"
#include "deep-learning/io/file.hpp"
#include "deep-learning/io/term/command.hpp"

// Standard:
#include <sstream>
#include <unistd.h>

using namespace DL::File;
using namespace DL::Str;
using namespace DL::Term;

namespace DL::Sys {
size_t avail_mem(void) {
  long const phys_page(sysconf(_SC_AVPHYS_PAGES)),
      page_size(sysconf(_SC_PAGE_SIZE));

  return static_cast<size_t>(phys_page) * static_cast<size_t>(page_size);
}

size_t total_mem(void) {
  long const phys_page(sysconf(_SC_PHYS_PAGES)),
      page_size(sysconf(_SC_PAGE_SIZE));

  return static_cast<size_t>(phys_page) * static_cast<size_t>(page_size);
}

std::vector<std::pair<std::wstring, std::wstring>> list_drives(void) {
  std::vector<std::pair<std::wstring, std::wstring>> drives;

  std::wistringstream output(check_output(L"cat /proc/mounts | grep /dev/sd"));

  std::wstring line, drive, mount, mount_home;

  size_t drive_at, mount_at;

  while (std::getline(output, line)) {
    if (line.find(L"/dev/sd") == std::wstring::npos) continue;

    drive_at = line.find_first_of(L" ");
    drive = line.substr(5_UZ, drive_at - 5_UZ);

    mount_at = line.find_first_of(L" ", drive_at + 1_UZ);
    mount = line.substr(drive_at + 1_UZ, mount_at - drive_at - 1_UZ);

    if (mount == L"/") {
      mount_home = mount + L"home/" + to_wstring(std::string(getlogin()));
      if (path_exist(mount_home)) mount = mount_home;
    }

    drives.push_back(std::pair<std::wstring, std::wstring>(drive, mount));
  }

  return drives;
}
}  // namespace DL::Sys
