/* Copyright 2016, 2019 S�bastien K�roack. All Rights Reserved.

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

namespace DL::Term {
class Keyboard {
 public:
  Keyboard(void);

  bool trigger_key(char const val);

  void clear_keys_pressed(void);
  void collect_keys_pressed(void);

 private:
  int _kbhit(void);

  std::wstring _map_chars;
};
}  // namespace DL::Term