# Compile definitions:
target_compile_definitions (${PROJECT_NAME} PRIVATE
  $<$<CONFIG:Debug>:
    "_DEBUG"
  >
  $<$<CONFIG:Release>:
    "NDEBUG"
  >
  "WIN32"
  "WIN64")
 
set_target_properties (${PROJECT_NAME} PROPERTIES
  VS_GLOBAL_KEYWORD "Win32Proj")

target_compile_options (${PROJECT_NAME} PRIVATE
  $<$<CONFIG:Debug>:
    "/fp:precise"
    "/MDd"
    "/Od"
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
  "/GR"
  "/GS-"
  "/nologo"
  "/permissive-"
  "/W3"
  "/Zc:forScope"
  "/Zc:inline"
  "/Zc:wchar_t"
  "/Zi")

target_link_options (${PROJECT_NAME} PRIVATE
  $<$<CONFIG:Debug>:
    "/INCREMENTAL"
    "/DEBUG"
  >
  $<$<CONFIG:Release>:
    "/INCREMENTAL:NO"
    "/LTCG:INCREMENTAL"
    "/OPT:ICF"
    "/OPT:REF"
  >
  "/DYNAMICBASE"
  "/ERRORREPORT:PROMPT"
  "/MANIFEST"
  "/MANIFESTUAC:\"level='asInvoker' uiAccess='false'\""
  "/MACHINE:X64"
  "/NXCOMPAT"
  "/NOLOGO"
  "/SUBSYSTEM:CONSOLE"
  "/TLBID:1")

# Adept:
target_include_directories(${PROJECT_NAME} PRIVATE
  $<$<CONFIG:Debug>:
    ${ADEPT_INCLUDE_DIR}
  >)

target_link_libraries (${PROJECT_NAME} PRIVATE
  $<$<CONFIG:Debug>:
    gtestd
    gtest_maind
  >
  $<$<CONFIG:Release>:
    gtest
    gtest_main
  >)
  