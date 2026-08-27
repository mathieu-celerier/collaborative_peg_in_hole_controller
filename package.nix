{
  lib,
  mkMcRtcController,
  cmake,
  mc-rtc,
  onnxruntime,
}:

mkMcRtcController {
  pname = "CollabPegInHoleController";
  version = "0.1.0";

  src = lib.cleanSource ./.;

  nativeBuildInputs = [ cmake ];
  # onnxruntime is propagated, not a plain buildInput, so it also lands in the devel shell's
  # inputsFrom: `../mcdev build` compiles src/OnnxBackend.cpp against the same headers the
  # release build uses, rather than against whatever a system copy happens to be.
  propagatedBuildInputs = [
    mc-rtc
    onnxruntime
  ];

  doCheck = false;

  passthru.mc-rtc = {
    controller = {
      Enabled = "CollabPegInHoleController";
      # Matches tests/etc/mc_rtc.yaml's MainRobot — the robot module this controller's
      # etc/CollabPegInHoleController.in.yaml is actually tuned for (KinovaBotaPegPlate, from
      # mc_kinova's topic/add-genA-bota branch, not mc_kinova's plain main).
      MainRobot = "KinovaBotaPegPlate";
    };
    # Not optional: etc/CollabPegInHoleController.in.yaml's `Plugins: [RosForceSensor]` is
    # loaded unconditionally at startup, and mc_rtc treats a missing plugin as a critical
    # error. Picked up by mc-rtc-nix's mkControllerSuperbuild via convertListToDrvsStrict,
    # which resolves these names against the package set assembled from
    # mc-rtc-kinova-external-forces's `packages` (see that flake's flake.nix).
    plugins = [
      "mc-ros-force-sensor"
    ];

    # Also not optional, and for the same reason: the ObserverPipelines block loads an
    # ExternalForcesObserver unconditionally, and a missing observer is fatal just like a
    # missing plugin. This is a fork of isri-aist/mc_external_forces_observer adding the
    # EF_Estimator::getResidualOnly / getFTSensorTorques / getExternalTorques datastore
    # getters, without which the person-A and person-B channels cannot be logged apart. It
    # replaces the mc-residual-estimation plugin that used to provide this.
    observers = [ "mc-external-forces-observer" ];

    # Picked up by mc-rtc-nix's mkControllerSuperbuild: with-suggested defaults to true, so
    # these get added to the generated superbuild shell automatically — mc-kinova's
    # passthru.mujocoRobots feeds mc-mujoco the matching robot XML (kinova_bota_peg_plate.xml).
    suggests = {
      apps = [ "mc-mujoco" ];
      robots = [ "mc-kinova" ];
    };

    # Packages we hack on from source rather than consuming from the nix store. These are only
    # consulted by the '-devel' shell, where mc-rtc-nix pulls in their *build* dependencies
    # (inputsFrom) and prepends $PWD/.superbuild/install to ControllerModulePaths /
    # RobotModulePaths / GlobalPluginPaths — so a local cmake install of these shadows the
    # nix-built copy with no flake edit, no hash, and no rebuild of the nix package.
    devel = {
      robots = [ "mc-kinova" ];
      observers = [ "mc-external-forces-observer" ];
    };
  };

  meta = with lib; {
    description = "Collaborative peg-in-hole compliance controller for mc_rtc";
    homepage = "https://github.com/mathieu-celerier/collaborative_peg_in_hole_controller";
    license = licenses.bsd2;
    platforms = platforms.all;
  };
}
