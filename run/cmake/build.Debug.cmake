# Compile definitions:
target_compile_definitions (${PROJECT_NAME} PRIVATE
  "_DEBUG")

if (CLANG OR GNU)
  target_compile_options (${PROJECT_NAME} PRIVATE
    "-fno-omit-frame-pointer"
    "-g2"
    "-gdwarf-2"
    "-O0")
else ()
  message (FATAL_ERROR "Unsupported compiler. Saw `${CMAKE_CXX_COMPILER}`")
endif ()