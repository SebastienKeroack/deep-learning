cmake_minimum_required (VERSION 3.14 FATAL_ERROR)

if (LINUX)
  set (PROJECT_NAME "deep-learning-nux")
elseif (WIN32)
  set (PROJECT_NAME "deep-learning-win")
endif ()

include ("cmake/all_files.cmake")

add_library (${PROJECT_NAME} STATIC ${ALL_FILES})
add_library (DL::DEEPLEARNING ALIAS ${PROJECT_NAME})

include ("cmake/increment_build_rev_ver.cmake")

# Project dir:
set (DEEPLEARNING_SUBDIR_LIB "${CMAKE_SOURCE_DIR}/lib")

include ("cmake/target_flags_wrt_compiler.cmake")

# Compile definitions:
target_compile_definitions (${PROJECT_NAME} PRIVATE
  "_LIB"
  "_UNICODE"
  "UNICODE"
  "X64")

# OS options:
if (LINUX)
  # systemd:
  find_package (PkgConfig REQUIRED)
  pkg_check_modules (SYSTEMD REQUIRED IMPORTED_TARGET libsystemd)
  target_link_libraries (${PROJECT_NAME} PRIVATE PkgConfig::SYSTEMD)
endif ()

# Output dir:
if (MSVC)
  set_target_properties (${PROJECT_NAME} PROPERTIES
    "OUTPUT_NAME_DEBUG"
      "deep-learning_${CMAKE_PROJECT_VERSION_H}_${CMAKE_PLATFORM}_d"
    "OUTPUT_NAME_RELEASE"
      "deep-learning_${CMAKE_PROJECT_VERSION_H}_${CMAKE_PLATFORM}")

  foreach (CONFIG ${CMAKE_CONFIGURATION_TYPES})
    string (TOUPPER "${CONFIG}" CONFIG_U)

    set_target_properties (${PROJECT_NAME} PROPERTIES
      "ARCHIVE_OUTPUT_DIRECTORY_${CONFIG_U}"
        "${CMAKE_BINARY_DIR}/${CMAKE_PLATFORM}/${CONFIG}")
  endforeach ()
else ()
  set_target_properties (${PROJECT_NAME} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/${CMAKE_PLATFORM}/${CMAKE_BUILD_TYPE}")
endif ()

target_compile_options (${PROJECT_NAME} PRIVATE
  ${OpenMP_CXX_COMPILE_FLAGS})

target_include_directories (${PROJECT_NAME} PRIVATE
  ${Boost_INCLUDE_DIR}
  ${DEEPLEARNING_SUBDIR_LIB}
  ${EIGEN3_INCLUDE_DIR}
  ${FMT_INCLUDE_DIR})

target_link_libraries (${PROJECT_NAME} PRIVATE
  fmt::fmt
  OpenMP::OpenMP_CXX)

include ("cmake/install.cmake")