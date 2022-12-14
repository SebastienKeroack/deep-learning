target_compile_options (${PROJECT_NAME} PRIVATE
  "-fexceptions"
  "-fno-strict-aliasing"
  "-frtti"
  "-fthreadsafe-statics"
  "-Wall")

target_link_options (${PROJECT_NAME} PRIVATE
  "ar"
  "-r'sc'")