cmake_minimum_required (VERSION 3.14 FATAL_ERROR)

if (LINUX)
  set (PROJECT_NAME "deep-learning-run-nux")
elseif (WIN32)
  set (PROJECT_NAME "deep-learning-run-win")
endif ()

include ("cmake/all_files.cmake")

add_executable (${PROJECT_NAME} ${ALL_FILES})

include ("cmake/increment_build_rev_ver.cmake")

# Solution dir:
set (DEEPLEARNING_ROOTDIR ${CMAKE_SOURCE_DIR})

# Deep learning lib dir:
set (DEEPLEARNING_SUBDIR_LIB "${CMAKE_SOURCE_DIR}/lib")

include ("cmake/target_flags_wrt_compiler.cmake")

# Compile definitions:
target_compile_definitions (${PROJECT_NAME} PRIVATE
  "_CONSOLE"
  "_UNICODE"
  "UNICODE"
  "X64")

# OS options:
if (LINUX)
  # Program suffix:
  set_target_properties (${PROJECT_NAME} PROPERTIES SUFFIX ".out")
endif ()

# Output dir:
if (MSVC)
  target_sources (${PROJECT_NAME} PRIVATE "deep-learning-run-win.rc")

  foreach (CONFIG ${CMAKE_CONFIGURATION_TYPES})
    string (TOUPPER "${CONFIG}" CONFIG_U)

    set_target_properties (${PROJECT_NAME} PROPERTIES
      "OUTPUT_NAME_${CONFIG_U}"
        "deep-learning_${CMAKE_PROJECT_VERSION_H}_${CMAKE_PLATFORM}"
      "RUNTIME_OUTPUT_DIRECTORY_${CONFIG_U}"
        "${CMAKE_BINARY_DIR}/${CMAKE_PLATFORM}/${CONFIG}")
  endforeach ()
else ()
  set_target_properties (${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME
      "deep-learning_${CMAKE_PROJECT_VERSION_H}_${CMAKE_PLATFORM}"
    RUNTIME_OUTPUT_DIRECTORY
      "${CMAKE_BINARY_DIR}/${CMAKE_PLATFORM}/${CMAKE_BUILD_TYPE}")
endif ()

target_include_directories (${PROJECT_NAME} PRIVATE
  ${DEEPLEARNING_ROOTDIR}
  ${DEEPLEARNING_SUBDIR_LIB}
  ${EIGEN3_INCLUDE_DIR})

target_link_libraries (${PROJECT_NAME} PRIVATE
  DL::DEEPLEARNING
  fmt::fmt)

# Compiler options:
if (CLANG OR GNU)
  target_link_options (${PROJECT_NAME} PRIVATE
    "-Wl,-z,now") # Full RELRO
endif ()