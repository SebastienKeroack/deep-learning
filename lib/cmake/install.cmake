include (CMakePackageConfigHelpers)

set (PROJECT_TARGETS_NAME "deep-learning-targets")
set (VERSION_CONF "${PROJECT_BINARY_DIR}/deep-learning-config-version.cmake")
set (PROJECT_CONF "${PROJECT_BINARY_DIR}/deep-learning-config.cmake")

set (DEEPLEARNING_INSTALL_INCLUDEDIR
  "include/deep-learning-${CMAKE_PROJECT_VERSION}")
set (DEEPLEARNING_INSTALL_DATADIR
  "share/deep-learning/cmake")

set (DEEPLEARNING_INCLUDE_DIR
  "${CMAKE_INSTALL_PREFIX}/${DEEPLEARNING_INSTALL_INCLUDEDIR}")
set (DEEPLEARNING_ROOT_DIR ${CMAKE_INSTALL_PREFIX})

# Install the library:
set_target_properties (${PROJECT_NAME} PROPERTIES EXPORT_NAME DEEPLEARNING)
install (TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_TARGETS_NAME})

# Install the headers:
install (
  DIRECTORY "deep-learning" DESTINATION ${DEEPLEARNING_INSTALL_INCLUDEDIR}
  FILES_MATCHING REGEX ".+\.(hpp|cuh)")

# Generate the version file.
configure_package_config_file (
  "cmake/deep-learning-config.cmake.in"
  ${PROJECT_CONF}
  PATH_VARS
    DEEPLEARNING_INCLUDE_DIR
    DEEPLEARNING_INSTALL_DATADIR
    DEEPLEARNING_ROOT_DIR
  INSTALL_DESTINATION ${DEEPLEARNING_INSTALL_DATADIR}
  NO_CHECK_REQUIRED_COMPONENTS_MACRO)
write_basic_package_version_file (
  ${VERSION_CONF}
  VERSION ${CMAKE_PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion)

# The DL target will be located in the DL namespace. Other CMake
# targets can refer to it using DL::DL.
export (TARGETS ${PROJECT_NAME} NAMESPACE DL::
  FILE "${PROJECT_TARGETS_NAME}.cmake")
# Export DL package to CMake registry such that it can be easily found by
# CMake even if it has not been installed to a standard directory:
export (PACKAGE DEEPLEARNING)

# Install version, config and target files:
install (
  FILES ${PROJECT_CONF} ${VERSION_CONF}
  DESTINATION ${DEEPLEARNING_INSTALL_DATADIR})
install (EXPORT ${PROJECT_TARGETS_NAME} NAMESPACE DL::
  DESTINATION ${DEEPLEARNING_INSTALL_DATADIR})