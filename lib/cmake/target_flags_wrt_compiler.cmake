# Compiler flags wrt build type except MSVC.
if (NOT MSVC)
  include ("cmake/build.${CMAKE_BUILD_TYPE}.cmake")
endif ()

# Compiler flags:
if (CLANG OR GNU)
  include ("cmake/clang_or_gnu.cmake")
elseif (MSVC)
  include ("cmake/msvc.cmake")
else ()
  message (FATAL_ERROR "Unsupported compiler. Saw `${CMAKE_CXX_COMPILER}`")
endif ()