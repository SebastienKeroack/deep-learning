# Compile definitions:
target_compile_definitions (${PROJECT_NAME} PRIVATE
  "NDEBUG")

if (CLANG OR GNU)
  target_compile_options (${PROJECT_NAME} PRIVATE
    "-fomit-frame-pointer"
    "-g1"
    "-march=native"
    "-O3")
else ()
  message (FATAL_ERROR "Unsupported compiler. Saw `${CMAKE_CXX_COMPILER}`")
endif ()

set_target_properties (${PROJECT_NAME} PROPERTIES
  OUTPUT_NAME
    "_deep-learning_${CMAKE_PROJECT_VERSION_H}_${CMAKE_PLATFORM}")