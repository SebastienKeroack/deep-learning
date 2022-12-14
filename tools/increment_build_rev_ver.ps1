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
$path = $args[0]

# Product/File suffix:
$PVFV = $args[1]

# version.hpp
$content = Get-Content -Raw $path -Encoding UTF8

# Product/File build version:
[int]$PVFV_BUILD_NEW = (Get-Date -UFormat %y) + (Get-Date -UFormat %j)
[int]$PVFV_BUILD=[regex]::Matches($content, "${PVFV}_BUILD (\d+)").Groups[1].value

# Product/File rev version:
[int]$PVFV_REV=[regex]::Matches($content, "${PVFV}_REV (\d+)").Groups[1].value

if ($PVFV_BUILD_NEW -ne $PVFV_BUILD) {
  $PVFV_BUILD = $PVFV_BUILD_NEW
  $PVFV_REV = 1
} else {
  $PVFV_REV += 1
}

# Write the new content:
$content = $content `
  -replace "${PVFV}_BUILD \d+", "${PVFV}_BUILD $PVFV_BUILD" `
  -replace "${PVFV}_REV \d+", "${PVFV}_REV $PVFV_REV" |
  Out-File -Encoding UTF8 -NoNewline $path