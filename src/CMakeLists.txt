# ONNXRUNTIME_INCLUDE_DIR / ONNXRUNTIME_LIB and the install rpath come from the top-level
# CMakeLists: the rpath has to be set before any target here is created.

set(controller_SRC CollabPegInHoleController.cpp PolicyPartner.cpp OnnxBackend.cpp
                   IntentReplay.cpp)

set(controller_HDR CollabPegInHoleController.h PolicyPartner.h PolicyBackend.h OnnxBackend.h
                   IntentReplay.h)

add_library(${PROJECT_NAME} SHARED ${controller_SRC} ${controller_HDR})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS
                                                 "-DCollabPegInHoleController_EXPORTS")
target_link_libraries(${PROJECT_NAME} PUBLIC mc_rtc::mc_control_fsm ${ONNXRUNTIME_LIB})
target_include_directories(${PROJECT_NAME} PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
install(
  TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION "${MC_RTC_LIBDIR}"
  LIBRARY DESTINATION "${MC_RTC_LIBDIR}"
  RUNTIME DESTINATION bin)

add_controller(${PROJECT_NAME}_controller lib.cpp "")
target_link_libraries(${PROJECT_NAME}_controller PUBLIC ${PROJECT_NAME})

add_subdirectory(states)
