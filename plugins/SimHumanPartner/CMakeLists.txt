# The plugin links the controller library because it drives the controller's own episode
# markers (startEpisode/markEpisode/stopEpisode) rather than inventing a second set. That
# keeps one definition of what an episode is, and lets scripts/extract_episodes.py slice
# simulated logs exactly as it slices real ones.
add_plugin(SimHumanPartner SimHumanPartner.cpp SimHumanPartner.h)
target_link_libraries(SimHumanPartner PUBLIC ${PROJECT_NAME})

install(FILES "etc/SimHumanPartner.yaml"
        DESTINATION "${MC_PLUGINS_RUNTIME_INSTALL_PREFIX}/etc")
