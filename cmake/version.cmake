# Get version from version.hpp:
set (PATH_VERSION_HPP "${CMAKE_SOURCE_DIR}/lib/version.hpp")
file (READ ${PATH_VERSION_HPP} content)

# Get major:
if (NOT content MATCHES "PV_MAJOR ([0-9]+)")
  message (FATAL_ERROR "Cannot get `PV_MAJOR` from `${PATH_VERSION_HPP}`.")
endif ()
set (CMAKE_PROJECT_VERSION_MAJOR ${CMAKE_MATCH_1})

# Get minor:
if (NOT content MATCHES "PV_MINOR ([0-9]+)")
  message (FATAL_ERROR "Cannot get `PV_MINOR` from `${PATH_VERSION_HPP}`.")
endif ()
set (CMAKE_PROJECT_VERSION_MINOR ${CMAKE_MATCH_1})

# Get build:
if (NOT content MATCHES "PV_BUILD ([0-9]+)")
  message (FATAL_ERROR "Cannot get `PV_BUILD` from `${PATH_VERSION_HPP}`.")
endif ()
set (CMAKE_PROJECT_VERSION_BUILD ${CMAKE_MATCH_1})

# Get rev:
if (NOT content MATCHES "PV_REV ([0-9]+)")
  message (FATAL_ERROR "Cannot get `PV_REV` from `${PATH_VERSION_HPP}`.")
endif ()
set (CMAKE_PROJECT_VERSION_REV ${CMAKE_MATCH_1})

# Concat M.m
set (CMAKE_PROJECT_VERSION
  "${CMAKE_PROJECT_VERSION_MAJOR}.${CMAKE_PROJECT_VERSION_MINOR}")
string (REPLACE "." "-" CMAKE_PROJECT_VERSION_H ${CMAKE_PROJECT_VERSION})