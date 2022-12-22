set (ALL_FILES
  $<$<CONFIG:Debug>:
    "deep-learning/array_grad_adept_test.cpp"
  >
  "deep-learning/array_grad_test.cpp"
  "pch.cpp"
  "pch.hpp")