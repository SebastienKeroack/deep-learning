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
#include "deep-learning/io/logger.hpp"

// Standard:
#include <tchar.h>
#include <windows.h>

#include <codecvt>

namespace DL::Sys {
size_t avail_mem(void) {
  MEMORYSTATUSEX mem_stats;

  mem_stats.dwLength = sizeof(MEMORYSTATUSEX);

  GlobalMemoryStatusEx(&mem_stats);

  return static_cast<size_t>(mem_stats.ullAvailPhys);
}

size_t total_mem(void) {
  MEMORYSTATUSEX mem_stats;

  mem_stats.dwLength = sizeof(MEMORYSTATUSEX);

  GlobalMemoryStatusEx(&mem_stats);

  return static_cast<size_t>(mem_stats.ullTotalPhys);
}

std::vector<std::pair<std::wstring, std::wstring>> list_drives(void) {
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
  std::vector<std::pair<std::wstring, std::wstring>> drives;

  std::wstring wdrive(L"A:");
  DWORD mask(GetLogicalDrives());

  if (mask == 0) {
    ERR(L"An error has been triggered from the "
        L"`GetLogicalDrives() -> %d` function.",
        GetLastError());
    return drives;
  }

  while (mask) {
    // Bitwise.
    if (mask & 1)
      drives.push_back(std::make_pair(wdrive, wdrive));

    // Increment the letter.
    ++wdrive[0];

    // Shift the bitmask binary right.
    mask >>= 1;
  }

  return drives;
}
}  // namespace DL::Sys
