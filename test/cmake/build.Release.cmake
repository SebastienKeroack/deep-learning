# Compile definitions:
target_compile_definitions (${PROJECT_NAME} PRIVATE
  "NDEBUG")

if (CLANG OR GNU)
  target_compile_options (${PROJECT_NAME} PRIVATE
    "-fomit-frame-pointer"
    "-g1"
    "-march=native"
    "-O3")

  target_link_options (${PROJECT_NAME} PRIVATE
    "-flto")
else ()
  message (FATAL_ERROR "Unsupported compiler. Saw `${CMAKE_CXX_COMPILER}`")
endif ()