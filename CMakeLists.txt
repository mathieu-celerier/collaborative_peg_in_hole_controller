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

# onnxruntime runs the RoboManipBaselines policies (src/OnnxBackend.cpp). Found here rather
# than in src/ because the rpath below has to reach every target: the FSM state libraries and
# the controller entry point all link libCollabPegInHoleController.so, and inherit its need for
# libonnxruntime.so.1.
#
# Not vendored the way rl_kinova_ball_balancing does it -- nix supplies it, and installing a
# second libonnxruntime.so beside the store's is a way to end up loading two of them.
find_path(
  ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  PATH_SUFFIXES onnxruntime
  REQUIRED)
find_library(
  ONNXRUNTIME_LIB
  NAMES onnxruntime
  REQUIRED)
get_filename_component(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_LIB}" DIRECTORY)

# Without this the install strips the build rpath and libonnxruntime.so.1 becomes
# unresolvable. The failure is silent and never mentions onnxruntime: mc_rtc dlopens each FSM
# state library, every dlopen fails because a dependency cannot be loaded, and what is reported
# is "origin state (..._Initial) is not loaded", then "None of the transitions you attempted to
# load in this TransitionMap are valid".
list(APPEND CMAKE_INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}")

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
