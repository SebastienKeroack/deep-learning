cmake_minimum_required (VERSION 3.14 FATAL_ERROR)

# Build type:
set (CMAKE_CONFIGURATION_TYPES
    "Debug"
    "Release"
    CACHE STRING "Alias /p:Configuration" FORCE)

include ("cmake/version.cmake")

# Solution name:
project ("deep-learning"
  VERSION ${CMAKE_PROJECT_VERSION}
  DESCRIPTION
    "Implementations of various advances made in the field of deep learning."
  HOMEPAGE_URL "https://github.com/SebastienKeroack/deep-learning"
  LANGUAGES "CXX")

# OS vars:
if (UNIX AND NOT APPLE)
  include ("cmake/nux.cmake")
elseif (WIN32)
  include ("cmake/win.cmake")
else ()
  message (FATAL_ERROR "Unsupported OS. Saw `${CMAKE_SYSTEM_NAME}`")
endif ()

set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags:
if (CLANG OR GNU)
  include ("cmake/clang_or_gnu.cmake")
elseif (MSVC)
  include ("cmake/msvc.cmake")
else ()
  message (FATAL_ERROR "Unsupported compiler. Saw `${CMAKE_CXX_COMPILER}`")
endif ()

# Boost:
find_package (Boost REQUIRED)

# Eigen:
find_package (Eigen3 CONFIG REQUIRED)
set (EIGEN3_INCLUDE_DIR "${EIGEN3_INCLUDE_DIR}/..")

# {fmt}:
find_package (fmt CONFIG REQUIRED)
set (FMT_INCLUDE_DIR "${fmt_DIR}/../../../include")

# OpenMP
find_package (OpenMP REQUIRED)

# Projects:
add_subdirectory ("lib")
add_subdirectory ("run")

if (NOT DEFINED NO_TESTS OR NO_TESTS)
  add_subdirectory ("test")
endif ()