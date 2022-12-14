set (PROJECT_VER_NAME "${PROJECT_NAME}-VER")

if (MSVC)
  add_custom_target(
    ${PROJECT_VER_NAME}
    COMMAND PowerShell -File
      "${CMAKE_SOURCE_DIR}/tools/increment_build_rev_ver.ps1"
      "${CMAKE_SOURCE_DIR}/test/version.hpp" "FV")
else ()
  add_custom_target(
    ${PROJECT_VER_NAME}
    COMMAND bash
      "${CMAKE_SOURCE_DIR}/tools/increment_build_rev_ver.sh"
      "${CMAKE_SOURCE_DIR}/test/version.hpp" "FV")
endif ()

add_dependencies(${PROJECT_NAME} ${PROJECT_VER_NAME})