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
#include "deep-learning-lib/pch.hpp"

// File header:
#include "deep-learning-lib/device/system/info.hpp"

// Deep learning lib:
#include "deep-learning-lib/io/logger.hpp"

namespace DL::Sys {
size_t remaining_mem(long double const reserved_bytes_pct,
                     size_t const maximum_reserved_bytes) {
  if (reserved_bytes_pct > 100.0L) {
    ERR(L"The number of reserved bytes in percent is greater than 100%%.");
    return 0_UZ;
  } else if (reserved_bytes_pct < 0.0L) {
    ERR(L"The number of reserved bytes in percent is less than 0%%.");
    return 0_UZ;
  }

  size_t const avail_bytes(avail_mem());
  size_t reserved_bytes(static_cast<size_t>(
      static_cast<long double>(total_mem()) * reserved_bytes_pct / 100.0L));

  reserved_bytes = std::min(reserved_bytes, maximum_reserved_bytes);

  return reserved_bytes > avail_bytes ? 0_UZ : avail_bytes - reserved_bytes;
}
}  // namespace DL::Sys