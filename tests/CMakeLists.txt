add_executable(test_newton test_newton.cc)
set_target_properties(test_newton PROPERTIES CXX_STANDARD 14)
set_target_properties(test_newton PROPERTIES CXX_STANDARD_REQUIRED ON)
target_link_libraries(test_newton parametrization_lib)

add_executable(test_svd_sensitivity test_svd_sensitivity.cc)
set_target_properties(test_svd_sensitivity PROPERTIES CXX_STANDARD 14)
set_target_properties(test_svd_sensitivity PROPERTIES CXX_STANDARD_REQUIRED ON)
target_link_libraries(test_svd_sensitivity parametrization_lib)

add_executable(test_eig_sensitivity test_eig_sensitivity.cc)
set_target_properties(test_eig_sensitivity PROPERTIES CXX_STANDARD 14)
set_target_properties(test_eig_sensitivity PROPERTIES CXX_STANDARD_REQUIRED ON)
target_link_libraries(test_eig_sensitivity parametrization_lib)

add_executable(test_balloon_energy test_balloon_energy.cc)
set_target_properties(test_balloon_energy PROPERTIES CXX_STANDARD 14)
set_target_properties(test_balloon_energy PROPERTIES CXX_STANDARD_REQUIRED ON)
target_link_libraries(test_balloon_energy parametrization_lib)

add_executable(test_tension_field_energy test_tension_field_energy.cc)
set_target_properties(test_tension_field_energy PROPERTIES CXX_STANDARD 14)
set_target_properties(test_tension_field_energy PROPERTIES CXX_STANDARD_REQUIRED ON)
target_link_libraries(test_tension_field_energy parametrization_lib)

################################################################################
# Catch2
################################################################################
add_executable(unit_tests
    catch2_main.cc
    catch2_svd_tests.cc
    catch2_tf_validation.cc
)

set_target_properties(unit_tests PROPERTIES CXX_STANDARD 14)
set_target_properties(unit_tests PROPERTIES CXX_STANDARD_REQUIRED ON)

target_link_libraries(unit_tests PUBLIC
	Catch2::Catch2
	parametrization_lib
	warnings::all
)

# Register tests
# Note: even though MeshFEM has added Catch2 to its CMAKE_MODULE_PATH,
# those changes are not visible to this parent project...
list(APPEND CMAKE_MODULE_PATH ${MESHFEM_EXTERNAL}/Catch2/contrib)
include(CTest)
include(Catch)
catch_discover_tests(unit_tests)
