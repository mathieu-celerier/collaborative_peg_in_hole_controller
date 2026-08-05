{
  description = "collaborative_peg_in_hole_controller: mc-rtc-superbuild release and development shells";

  inputs = {
    mc-rtc-nix.url = "github:mc-rtc/nixpkgs";
    flake-parts.follows = "mc-rtc-nix/flake-parts";
    systems.follows = "mc-rtc-nix/systems";

    mc-rtc-kinova-external-forces.url = "github:mathieu-celerier/mc-rtc-kinova-external-forces";

    ccache-trigger.url = "github:boolean-option/true";
  };

  outputs =
    inputs:
    inputs.mc-rtc-nix.lib.mkMcRtcController inputs "CollabPegInHoleController" (
      { ... }:
      {
        imports = [
          inputs.mc-rtc-kinova-external-forces.flakeModule

          # Puts mc_rtc itself in the devel shell's `inputsFrom`, so the shell carries mc_rtc's
          # own build dependencies and it can be built from a local checkout into
          # .superbuild/install (where LD_LIBRARY_PATH then shadows the nix-built copy).
          #
          # This CANNOT go in package.nix's passthru.mc-rtc.devel: mkControllerSuperbuild only
          # maps `robots`, `plugins`, `controllers` and `config` out of that block, so a
          # `devel.apps` there is silently dropped. It also cannot go in this module's body:
          # mkMcRtcController does `body // defaultAttrs`, which would clobber a top-level
          # `mc-rtc-superbuild` key. As an `imports` entry it survives, and since
          # mc-rtc-superbuild is a deferredModule it merges with the auto-generated
          # CollabPegInHoleController configuration rather than replacing it.
          {
            mc-rtc-superbuild =
              { pkgs, lib, ... }:
              let
                # Entries can be plain paths (setup hooks), not just derivations, so match on
                # the string form rather than assuming a `name` attribute exists.
                dropQt =
                  p:
                  let
                    n = if builtins.isAttrs p then (p.name or "") else builtins.baseNameOf p;
                  in
                  !(lib.hasInfix "qtbase" n || lib.hasInfix "wrap-qt" n);
              in
              {
                # 1 kHz, matching ~/.config/mc_rtc/mc_rtc.yaml (Timestep: 0.001) which the
                # superbuild picks up but this module cannot: superbuild.nix:181 writes
                # `LoadUserConfiguration: false` into the generated .superbuild/mc_rtc.yaml, so
                # the user config is ignored and mc_rtc falls back to 5 ms.
                #
                # This is not a preference, it is a stability requirement. In the Compliant
                # state the posture task runs at stiffness 0, leaving the arm's 1-DoF null-space
                # self-motion regulated by damping alone. At dt=5 ms that mode is unstable: the
                # logs show it growing out of numerical noise at ~x1.10 per step, alternating
                # sign every tick, until alphaDOut hits +/-127 rad/s^2 while the end-effector
                # moves 0.4 mm. At dt=1 ms the same continuous-time mode is comfortably stable —
                # the working superbuild run peaks at |alphaDOut| = 16 over 26 s.
                configurations.CollabPegInHoleController.timeStep = 0.001;

                configurations.CollabPegInHoleController.devel.apps = [
                  # This derivation is never built: `inputsFrom` only reads its dependency
                  # lists to assemble the shell. So the two adjustments below cost nothing and
                  # do not affect the mc-rtc that actually ends up in the shell and on the
                  # runtime module paths — they only shape what the shell inherits.
                  (pkgs.mc-rtc.overrideAttrs (prev: {
                    # separateDebugInfo = true makes stdenv append the bare *path*
                    # build-support/setup-hooks/separate-debug-info.sh to nativeBuildInputs.
                    # make-shell types that option as `listOf package`, and a plain path fails
                    # the check ("is not of type `package'"). mc-kinova and
                    # mc-residual-estimation don't set it, which is why only mc-rtc trips this.
                    separateDebugInfo = false;

                    # mc-rtc pulls in qt.qtbase and wrapQtAppsHook (Qt5 here, since mc-rtc-nix
                    # derives qtVersion from the ROS distro), while the shell's ros-env brings
                    # Qt6. Both landing in the shell makes the Qt setup hook abort with
                    # "detected mismatched Qt dependencies" — the hook must go too, not just
                    # qtbase, since it propagates qtbase-dev by itself. Dropping them is safe:
                    # mc_rtc's CMakeLists never references Qt at all — these are here only for
                    # the PyQt5-based utils (mc_rtc_log_ui and friends), which the store-built
                    # mc-rtc in this same shell already provides.
                    buildInputs = lib.filter (dropQt) prev.buildInputs;
                    nativeBuildInputs = lib.filter (dropQt) prev.nativeBuildInputs;
                  }))
                ];
              };
          }
        ];

        mc-rtc-nix.overlays.ccache = inputs.ccache-trigger.value;

        flakoboros.packages.CollabPegInHoleController = ./package.nix;
      }
    );
}
