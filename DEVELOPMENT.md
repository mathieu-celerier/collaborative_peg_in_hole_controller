# Developing against a local mc_rtc

This controller consumes mc_rtc, mc_kinova and the plugins from the nix store by default. To
edit any of them, keep a checkout as a sibling of this repo:

```
~/devel/mc-rtc-nix/
  collaborative_peg_in_hole_controller/   <- you are here
  dev-overrides.sh                        <- machine-local, untracked
  mc_rtc/
  mc_kinova/
  mc_residual_estimation/
  MinimumJerkTask/
  mc_kortex/
```

`../dev-overrides.sh` points the flake inputs at whichever of those directories exist, using
`--override-input`. Nothing in a committed flake depends on this: delete the file (or clone this
repo somewhere else) and everything falls back to the pinned `github:` sources.

## Two loops

There are two ways to get a local change into a running controller. They read the **same**
checkout, so you can move between them freely.

### Fast (~20 s) — for `.cpp` changes

Build the package into `.superbuild/install`. The devel shell prepends that to
`LD_LIBRARY_PATH`, and because nix links with `DT_RUNPATH` (which `LD_LIBRARY_PATH` overrides),
even store-built consumers load your library instead of the store one.

```sh
cd ~/devel/mc-rtc-nix/mc_rtc
../mcdev build
```

**Use `../mcdev`, not a bare `cmake`.** The core checkouts are *siblings* of this controller,
not children, so direnv unloads the devel environment the instant you `cd` into one — leaving
`$cmakeFlags` and `$INSTALL_DIR` empty. mc_rtc then fails at configure with:

```
CMake Error at CMakeLists.txt:401 (install):  install DIRECTORY given no DESTINATION!
CMake Error at CMakeLists.txt:413 (file):     file RELATIVE_PATH must be passed a full path
```

both of which are just `${CMAKE_INSTALL_PREFIX}` expanding to nothing. (The old superbuild never
hit this because sources lived under `.superbuild/`, inside the direnv tree.) `mcdev` wraps
`direnv exec`, which loads the environment while keeping your working directory.

Do **not** "fix" this by dropping an `.envrc` into a checkout: mc_rtc already tracks one, and
because the flake inputs are `git+file://`, editing any tracked file changes the source hash —
you would silently get a different mc_rtc in the shell than `nix build` produces.

`mcdev build` picks `$cmakeFlags` for mc_rtc and `$CMAKE_LOCAL_FLAGS` for everything else, and
adds `-DINSTALL_DOCUMENTATION=OFF` — otherwise doxygen runs on every build and copies read-only
MathJax files out of the store, after which a plain `rm -rf build` fails with "Permission
denied" (`chmod -R u+w build` first if you hit it).

Other forms: `mcdev` alone opens a subshell with the environment; `mcdev <cmd>` runs anything
else in it.

Header resolution in the shell is subtle and `.envrc` handles it for you: the superbuild
shellHook clears `NIX_CFLAGS_COMPILE` so store mc_rtc headers cannot shadow your local ones, and
nixpkgs only copies `NIXPKGS_CMAKE_PREFIX_PATH` into `CMAKE_PREFIX_PATH` during
`cmakeConfigurePhase`, which a dev shell never runs. `.envrc` therefore sets
`CMAKE_PREFIX_PATH="$INSTALL_DIR:$NIXPKGS_CMAKE_PREFIX_PATH"`. Without it, mc_rtc fails to
compile with e.g. `fatal error: state-observation/dynamics-estimators/lipm-dcm-estimator.hpp:
No such file or directory`.

No `direnv reload`, no nix rebuild. Verify what actually got loaded with:

```sh
LD_DEBUG=libs mc_mujoco 2>&1 | grep libmc_control
```

### Sync (~3–5 min) — when you are done, or when you changed a header

```sh
direnv reload
```

Nix rebuilds mc_rtc and everything downstream from the same checkout. ccache
(`/var/cache/ccache`) keeps this to minutes: for a `.cpp`-only change, dependents preprocess
identically, hit the cache, and only re-link.

## When you MUST use the sync loop

**Any change to an mc_rtc header, or anything else that changes ABI.**

The fast loop only rebuilds what is listed in `package.nix`'s `passthru.mc-rtc.devel` (plus
mc_rtc, wired up in `flake.nix`). Everything else — `mc-mujoco`, `mc-kortex`,
`mc-joystick-plugin`, `mc-ros-force-sensor` — still comes from the store, still linked against
the *old* ABI. Mixing them gives silent memory corruption or a crash at controller load, not a
helpful link error.

Rule of thumb: touched only `.cpp` bodies → fast loop is fine. Touched a header, a struct
layout, a virtual, or a template → `direnv reload`.

If you want a package to participate in the fast loop, add it to `devel` in `package.nix`:

```nix
devel = {
  robots = [ "mc-kinova" ];
  plugins = [ "mc-residual-estimation" ];
};
```

## Gotcha: new files need `git add -N`

The overrides use `git+file://`, which respects `.gitignore` (this is deliberate — `path:` would
copy `build/` into the store on every compile). The trade-off is that nix only sees files git
knows about. After creating a new source file:

```sh
git add -N src/MyNewState.cpp
```

Otherwise the sync build fails with CMake complaining about a missing file that is plainly
there.

## Gotcha: the devel shell is NOT rebuilt by the controller being rebuilt

`.superbuild/mc_rtc.yaml` lists **only** `.superbuild/install/lib{,64}/mc_controller` under
`ControllerModulePaths` — unlike robots and plugins, the controller under development has no nix
store fallback. Deleting `.superbuild/install` therefore gives `no controller selected`, not a
fallback to the store copy. Rebuild it with the fast loop.

Note also that `nix build .#CollabPegInHoleController` produces a controller that is *not* on any
module path in devel mode. To test the pure-nix build, use the release shell instead.

## Gotcha: nix-direnv caches across `--override-input` changes

nix-direnv keys its cache on the flake attribute alone, and refreshes the profile's mtime on
every load, so neither adding/removing overrides nor editing `../dev-overrides.sh` invalidates
it. Left alone, the shell serves whichever mc_rtc was current when the cache was first built,
while `nix build` uses the overridden one — two different ABIs in one session.

`.envrc` now stamps `MC_RTC_DEV_OVERRIDES` into `.direnv/dev-overrides.stamp` and drops
`.direnv/flake-profile-*` when it changes. If you ever suspect staleness anyway:

```sh
rm -rf .direnv && nix-direnv-reload
```

`nix-direnv-reload` (not plain `direnv reload`) is what actually rebuilds the profile under
`nix_direnv_manual_reload`; it is a shell function, so it only exists in an interactive shell.

## Which mc_rtc am I actually running?

```sh
echo $MC_RTC_CONTROLLER_CONFIG      # .superbuild/mc_rtc.yaml comes first in devel mode
nix eval .#packages.x86_64-linux.CollabPegInHoleController.outPath
ls $INSTALL_DIR/lib $INSTALL_DIR/lib64   # what the fast loop has shadowing the store
```
