# Set OS boolean:
set (LINUX TRUE)

# Build type:
if (NOT CMAKE_BUILD_TYPE IN_LIST CMAKE_CONFIGURATION_TYPES)
  message (FATAL_ERROR "Build type '${CMAKE_BUILD_TYPE}' invalid. "
    "Expected one of: ${CMAKE_CONFIGURATION_TYPES}")
endif ()

# Compiler:
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set (CLANG TRUE)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set (GNU TRUE)
else ()
  message (FATAL_ERROR "Unsupported compiler. Saw `${CMAKE_CXX_COMPILER_ID}`")
endif ()