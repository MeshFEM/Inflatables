################################################################################
# Stripe-patterns-based wall generation library
################################################################################
add_library(wallgen_lib evaluate_stripe_field.cc extract_contours.cc)
set_target_properties     (wallgen_lib PROPERTIES CXX_STANDARD 14)
set_target_properties     (wallgen_lib PROPERTIES CXX_STANDARD_REQUIRED ON)
target_include_directories(wallgen_lib PUBLIC .)
target_link_libraries     (wallgen_lib MeshFEM StripePatterns)
