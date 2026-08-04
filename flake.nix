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
        imports = [ inputs.mc-rtc-kinova-external-forces.flakeModule ];

        mc-rtc-nix.overlays.ccache = inputs.ccache-trigger.value;

        flakoboros.packages.CollabPegInHoleController = ./package.nix;
      }
    );
}
