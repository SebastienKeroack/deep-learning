#!/bin/bash
# Copyright 2022 Sébastien Kéroack. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Path to the version file:
declare -r path=$1

# Product/File suffix:
declare -r PVFV=$2

# version.hpp
content=$(tr -d '\0' < $path)

# Product/File build version:
declare -i PVFV_BUILD_NEW="$(date +%y)$(date +%j)"
[[ $content =~ "${PVFV}_BUILD "([0-9]+) ]]
declare -i PVFV_BUILD=${BASH_REMATCH[1]}

# Product/File rev version:
[[ $content =~ "${PVFV}_REV "([0-9]+) ]]
declare -i PVFV_REV=${BASH_REMATCH[1]}

if [[ $PVFV_BUILD_NEW -ne $PVFV_BUILD ]]; then
  ((PVFV_BUILD = PVFV_BUILD_NEW))
  ((PVFV_REV = 1))
else
  ((++PVFV_REV))
fi

content=$(
  sed -E "s/${PVFV}_BUILD [0-9]+/${PVFV}_BUILD ${PVFV_BUILD}/g" <<< $content)
content=$(
  sed -E "s/${PVFV}_REV [0-9]+/${PVFV}_REV ${PVFV_REV}/g" <<< $content)

# Write the new content:
echo "$content" > $path