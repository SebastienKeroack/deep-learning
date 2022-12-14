# Set platform name:
set (CMAKE_PLATFORM ${CMAKE_HOST_SYSTEM_PROCESSOR})
string (REPLACE "x86_64" "x64" CMAKE_PLATFORM ${CMAKE_PLATFORM})

# Set default OpenMP flags:
set (OpenMP_CXX_COMPILE_FLAGS "-fopenmp")
set (OpenMP_EXE_LINKER_FLAGS "-fopenmp=libiomp5")