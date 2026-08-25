cmake_minimum_required(VERSION 3.22)

set(CXX_DISABLE_WERROR 1)
set(CMAKE_CXX_STANDARD 11)

set(PROJECT_NAME CollabPegInHoleController)
set(PROJECT_DESCRIPTION "CollabPegInHoleController")
set(PROJECT_URL "")

project(
  ${PROJECT_NAME}
  LANGUAGES CXX
  VERSION 1.0.0)
include(CTest)
enable_testing()

if(NOT TARGET mc_rtc::mc_control)
  find_package(mc_rtc REQUIRED)
endif()

add_subdirectory(src)

# Simulation-only: applies the simulated human partners as external wrenches in mc_mujoco.
# Built unconditionally (it is inert without mc_mujoco's ApplyWrenchOnBody datastore call)
# but never loaded on hardware -- Plugins in etc/CollabPegInHoleController.yaml decides.
add_subdirectory(plugins/SimHumanPartner)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

configure_file(etc/CollabPegInHoleController.in.yaml
               "${CMAKE_CURRENT_BINARY_DIR}/etc/CollabPegInHoleController.yaml")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/etc/CollabPegInHoleController.yaml"
        DESTINATION "${MC_CONTROLLER_RUNTIME_INSTALL_PREFIX}/etc")

# Per-controller plugin config override, read by mc_rtc from
# etc/CollabPegInHoleController/plugins/<Plugin>.yaml under this install prefix (see
# MCGlobalController::GlobalConfiguration::load_controller_plugin_configs()).
install(FILES "etc/CollabPegInHoleController/plugins/RosForceSensor.yaml"
        DESTINATION "${MC_CONTROLLER_RUNTIME_INSTALL_PREFIX}/etc/CollabPegInHoleController/plugins")
