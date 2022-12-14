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
#include <string>
#include <vector>

namespace DL::Sys {
// Type return: The amount of physical memory currently available, in bytes.
// This is the amount of physical memory that can be immediately reused without
// having to write its contents to disk first. It is the sum of the size of the
// standby, free, and zero lists.
size_t avail_mem(void);

size_t remaining_mem(long double const reserved_bytes_pct,
                     size_t const maximum_reserved_bytes);

// Type return: The amount of actual physical memory, in bytes.
size_t total_mem(void);

std::vector<std::pair<std::wstring, std::wstring>> list_drives(void);
}  // namespace DL::Sys