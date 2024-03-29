﻿# Avoid warning about `DOWNLOAD_EXTRACT_TIMESTAMP` in CMake `3.24`:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy (SET CMP0135 NEW)
endif()

include (FetchContent)
FetchContent_Declare (googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.12.1)

FetchContent_MakeAvailable (googletest)