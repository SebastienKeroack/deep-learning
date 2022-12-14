# Compile definitions:
target_compile_definitions (${PROJECT_NAME} PRIVATE
  $<$<CONFIG:Debug>:
    "_DEBUG"
    "ADEPT_RECORDING_PAUSABLE"
    "BOOST_SPIRIT_X3_DEBUG"
  >
  $<$<CONFIG:Release>:
    "NDEBUG"
  >
  "WIN32"
  "WIN64")

set_target_properties (${PROJECT_NAME} PROPERTIES
    VS_GLOBAL_KEYWORD "Win32Proj")

target_compile_definitions (${PROJECT_NAME} PRIVATE
  "_SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING")

target_compile_options (${PROJECT_NAME} PRIVATE
  $<$<CONFIG:Debug>:
    "/fp:precise"
    "/MDd"
    "/Od"
    "/permissive"
    "/RTC1"
  >
  $<$<CONFIG:Release>:
    "/fp:fast"
    "/MD"
    "/O2"
    "/Oi"
    "/GL"
    "/Gy"
  >
  "/diagnostics:column"
  "/EHsc"
  "/errorReport:prompt"
  "/FC"
  "/Gd"
  "/Gm-"
  "/GS-"
  "/nologo"
  "/W3"
  "/Zc:forScope"
  "/Zc:inline"
  "/Zc:wchar_t"
  "/Zi")

target_link_options (${PROJECT_NAME} PRIVATE
  $<$<CONFIG:Release>:
    "/INCREMENTAL:NO"
    "/LTCG"
  >
  "/MACHINE:X64"
  "/NOLOGO")

# Adept:
target_include_directories(${PROJECT_NAME} PRIVATE
  $<$<CONFIG:Debug>:
    ${ADEPT_INCLUDE_DIR}
  >)