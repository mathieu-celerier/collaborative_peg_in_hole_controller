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
    };
  };

  meta = with lib; {
    description = "Collaborative peg-in-hole compliance controller for mc_rtc";
    homepage = "https://github.com/mathieu-celerier/collaborative_peg_in_hole_controller";
    license = licenses.bsd2;
    platforms = platforms.all;
  };
}
