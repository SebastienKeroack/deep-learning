# Compile definitions:
target_compile_definitions (${PROJECT_NAME} PRIVATE
  "_DEBUG"
  "ADEPT_RECORDING_PAUSABLE"
  "BOOST_SPIRIT_X3_DEBUG")

if (CLANG OR GNU)
  target_compile_options (${PROJECT_NAME} PRIVATE
    "-fno-omit-frame-pointer"
    "-g2"
    "-gdwarf-2"
    "-O0")
else ()
  message (FATAL_ERROR "Unsupported compiler. Saw `${CMAKE_CXX_COMPILER}`")
endif ()

set_target_properties (${PROJECT_NAME} PROPERTIES
  OUTPUT_NAME
    "_deep-learning_${CMAKE_PROJECT_VERSION_H}_${CMAKE_PLATFORM}_d-a")

# Adept
target_link_libraries (${PROJECT_NAME} PRIVATE
  adept)