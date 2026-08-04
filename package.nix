{
  lib,
  mkMcRtcController,
  cmake,
  mc-rtc,
}:

mkMcRtcController {
  pname = "CollabPegInHoleController";
  version = "0.1.0";

  src = lib.cleanSource ./.;

  nativeBuildInputs = [ cmake ];
  propagatedBuildInputs = [ mc-rtc ];

  doCheck = false;

  passthru.mc-rtc = {
    controller = {
      Enabled = "CollabPegInHoleController";
      # Matches tests/etc/mc_rtc.yaml's MainRobot — the robot module this controller's
      # etc/CollabPegInHoleController.in.yaml is actually tuned for (KinovaBotaPegPlate, from
      # mc_kinova's topic/add-genA-bota branch, not mc_kinova's plain main).
      MainRobot = "KinovaBotaPegPlate";
    };
    # Not optional: etc/CollabPegInHoleController.in.yaml's `Plugins: [RosForceSensor,
    # ExternalForcesEstimator]` is loaded unconditionally at startup, and mc_rtc treats a
    # missing plugin as a critical error. Picked up by mc-rtc-nix's mkControllerSuperbuild via
    # convertListToDrvsStrict, which resolves these names against the package set assembled
    # from mc-rtc-kinova-external-forces's `packages` (see that flake's flake.nix).
    plugins = [
      "mc-ros-force-sensor"
      "mc-residual-estimation"
    ];

    # Picked up by mc-rtc-nix's mkControllerSuperbuild: with-suggested defaults to true, so
    # these get added to the generated superbuild shell automatically — mc-kinova's
    # passthru.mujocoRobots feeds mc-mujoco the matching robot XML (kinova_bota_peg_plate.xml).
    suggests = {
      apps = [ "mc-mujoco" ];
      robots = [ "mc-kinova" ];
    };
  };

  meta = with lib; {
    description = "Collaborative peg-in-hole compliance controller for mc_rtc";
    homepage = "https://github.com/mathieu-celerier/collaborative_peg_in_hole_controller";
    license = licenses.bsd2;
    platforms = platforms.all;
  };
}
