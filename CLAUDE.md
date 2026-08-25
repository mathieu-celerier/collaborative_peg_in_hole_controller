# CollabPegInHoleController — project context

## What this is for

An mc_rtc controller for **physical human–human collaborative peg-in-hole insertion** on a Kinova
arm, built to **collect demonstrations for training an imitation-learning policy**. The recording
pipeline is the deliverable, not a side feature.

Two people work on the robot at the same time, on opposite sides of the F/T sensor:

| | Who | Acts on | Observed through |
|---|---|---|---|
| **Person A** | the *demonstrator* | the robot **body**, upstream of the F/T sensor | the joint-torque **residual** (`ExternalForcesObserver`) — the F/T sensor cannot see them |
| **Person B** | the *collaborator* | the **piece** held at the end effector, downstream of the sensor | the **F/T sensor** wrench (`EEForceSensor`, Bota, fed by `RosForceSensor` on the real robot) |

Person B holds/guides the piece; person A operates the robot by hand-guiding its body so the piece
reaches the peg. The end goal is to **train a policy that replaces person A**: the robot takes over
the demonstrator's role and performs the insertion in collaboration with person B. That is why the
two interaction channels are kept separate in the logs — the policy's inputs come from person B's
channel plus the robot state, and its targets come from what person A did.

`EF_Estimator::getResidualOnly` / `getFTSensorTorques` / `getExternalTorques` come from a fork,
`mathieu-celerier/mc_external_forces_observer@topic/residual-datastore-accessors`: upstream
(`isri-aist`) logs all three signals but exposes none of them on the datastore. The fork adds
only those three getters. Do not push to `isri-aist`.

The `collab_ext_wrench` entry is the only signal that sees *both* people: it maps the estimator's
fused external joint torques back to a task-space wrench at the tool frame.

## Controller shape

Two-state FSM (`src/states/`), switched from the GUI (`{Collab, Mode}`), not by FSM transitions:

- **Initial** — stiff, position-controlled homing/setup; estimator feedback off.
- **Compliant** — torque control; both people can hand-guide simultaneously. `updateDualCompliance()`
  runs every tick: above `dual_force_enter` on the (smoothed) trigger wrench the end-effector
  stiffness ramps to ~0 so the piece floats; below `dual_force_exit` it re-stiffens and holds
  (hysteresis + low-pass + moving average + gain ramp, to stop chatter).

## Recording workflow

GUI tab `{Collab, Episode}`: **Start episode** → perform the insertion → **Mark success/failure** →
**Stop episode**. This writes markers into the ordinary mc_rtc session log
(`collab_recording`, `collab_episode_id`, `collab_episode_outcome`, `collab_episode_tick`); the
outcome is *sticky* for the whole episode. `scripts/extract_episodes.py` slices the log on those
markers into one `.npz` per demonstration. See `README.md` for the operator-facing version.

Two things that do **not** work for per-episode log files, both verified by trying them:
- `Logger::open(file, dt, t)` emits no `KeyAddedEvent` for already-registered entries → the
  resulting `.bin` has data rows with no schema and **segfaults** `mc_bin_utils` / `mc_bin_to_flat`.
- `Logger::start(name, dt, resume=true)` re-declares the entries, but names the file from a
  **second-resolution** timestamp → a stop-then-start inside the same second silently truncates
  the episode that just finished.

Hence: one session log, segmented by markers.

## Gotchas worth remembering

- The per-robot config block is keyed on `robot().module().name` — `kinova_bota_peg_plate`, the
  snake_case variant string from mc_kinova, **not** the `MainRobotParams` name
  `KinovaBotaPegPlate`. A wrong key is skipped in silence and every value reverts to its C++
  default.
- Plugin config overrides must live in `etc/CollabPegInHoleController/plugins/<Plugin>.yaml`; a
  top-level key in the controller yaml is never read for plugin init.
- Without the `ExternalForcesObserver` there is no person-A channel at all and the compliant
  tasks get no force feedback (the controller warns once). It is an **observer**, configured
  inline in `ObserverPipelines`, not a global plugin — it replaced `mc_residual_estimation`'s
  `ExternalForcesEstimator`, keeping the same `EF_Estimator::` datastore prefix. The residual it
  reports is person A *alone*: the F/T torques enter its momentum integral as a known input, so
  person B is subtracted rather than mixed in. It also has no `ft_sensor_name` — it projects
  every force sensor on the robot.
- **The F/T calibration is not interchangeable between the robot and the simulator**, and the
  wrong one silently disables gravity compensation.
  `applySimForceSensorCalibration()` handles it: called from `reset()`, it probes for
  `"<robot>::SetPDGains"` — which mc_mujoco registers in `MjSimImpl::makeDatastoreCalls()`,
  before `controller->init()` and so before `reset()`, and which mc_kortex never registers —
  and on a hit reloads the calibrator on **both** `robot()` and `realRobot()` (the observer
  projects the sensors of the latter) from mc_kinova's `calib/sim/<variant>/`. No-op on
  hardware. A hardware calibration folds in the Bota's force gain (0.9633)
  and its tare — the offset is minus the model wrench at the home pose, because the physical
  sensor is zeroed there. MuJoCo's site sensors have neither: they are exact and untared,
  reporting the full subtree reaction (the plate's 6.5 N) at every pose. Applied in simulation
  the gain and tare errors cancel *at the home pose* and nowhere else, so
  `wrenchWithoutGravity()` returns ~the raw wrench. mc_kinova ships both: `calib/<variant>/`
  (hardware, what the module loads) and `calib/sim/<variant>/` (gain 1, offset 0, mass/CoM from
  the URDF). `MC_KINOVA_CALIB=sim` forces the simulation one for consumers that do not
  auto-detect. See mc_kinova `calib/README.md`.
  Check: `collab_ft_wrench_norm_nograv` must read ~0 at rest, not ~6.6, **at more than one pose**
  — the wrong calibration zeroes at the home pose only, which is exactly where you would think
  to look.
- An uncompensated F/T reading is invisible until the observer resets, which is exactly what
  entering Compliant does. The F/T projection is fed to the momentum observer as a known input, so the residual
  quietly converges to cancel the phantom and the total stays ~0. `setEstimatorActive(true)` →
  `EF_Estimator::toggleActive` → `resetObserver_`, which zeroes the residual **but not the F/T
  term** — so `tau_ext_hat` steps by the full phantom (~4 Nm) into the QP in one tick and the arm
  diverges within ~25 ms. This is why it only ever exploded with `use_forces_from_ft_sensors: true`.
  The calibration removes the phantom, but the discontinuity itself is still there: any real F/T
  reading at the instant of the reset — person B already holding the plate, or hardware bias on the
  real Bota — reproduces it. The fix, if it bites: in the fork's `resetMomentumObserver()`,
  pre-load `integralTerm_ = tau_ft / residualGain_` and `tau_momentum_observer_ = -tau_ft` so
  `tau_ext_hat_` is continuous across the reset, as `pZero_` already does for the momentum term.
- **`tau_ext` and the robot's own motion form an algebraic loop through the payload, and the
  freed end-effector rings on it at ~90 Hz.** This is the failure behind "it explodes as soon as
  the force is strong enough to trigger the compliance", and it needs no hard push at all.
  The F/T sensor reports the reaction of everything distal to it, and `wrenchWithoutGravity()`
  removes gravity, **not inertia**. The plate's CoM is 139 mm off the sensor, so tool angular
  acceleration alone appears as `m*|alpha x r|` ~ `0.095*alpha` N of pure artefact -- 100 rad/s^2
  is ~10 N. The compliant task feeds `tau_ext` straight back as commanded acceleration
  (`refAccel = J H^-1 tau_ext`), so sensor -> estimate -> command -> motion -> sensor closes
  **within one tick**; once the end-effector stiffness ramps down, nothing dissipates it.
  Measured, 2026-08-24 19:30 run: operator holding a steady 12 N, `|alphaIn|` 0.07, flat for
  seconds; then at the free edge (tick 5624) joints 1, 5 and 7 ring with an ~11 ms period growing
  ~3x per cycle, while **the tool travels 1.3 mm and the F/T force swings +45 to -77 N**. QP
  infeasible 26 ms later, `QACC at DOF 6` (= joint_1). `tauMomentumObserver` stays at ~0.02 N.m
  throughout, so none of it is person A -- the ring is entirely in the F/T channel, which is why
  it only appears with `use_forces_from_ft_sensors: true` and why it looked like the calibration
  bug. `collab_tau_ext_clamped_dofs` was 0: no torque limit was involved.
  The cure is `conditionExternalTorques()`'s first-order low-pass on `tau_ext` into the solver
  (`external_torque_filter_hz`, 10 Hz). A person guiding an arm lives below ~5 Hz, so this keeps
  the human channel and removes an order of magnitude at the ring. Recorded signals are read
  before it runs, so the dataset stays raw.
  It also makes `ee_orientation_compliant: false` viable, which is the setting the work needs:
  defending the orientation means commanding torque *against* `tau_ext`, which is precisely what
  excites the loop. If it rings again, lower the cutoff before touching the compliance vector.
  The principled complement, still to do: subtract the payload's dynamic reaction from the wrench
  using the commanded `alphaD` and the distal spatial inertia, so the sensor reports what the
  human applies rather than what the robot did. The filter is not a substitute for that -- but a
  model-based compensation is only as good as the model, and this loop needs no gain margin.

- **A large external-force estimate does not overload the arm, it makes the QP *infeasible*.**
  `DynamicsConstraint` is built with `compensateExtTorques=true`, so the estimator's `tau_ext`
  enters the motion constraint as a known term of `H qdd + C = tau + tau_ext` while `|tau|` stays
  bounded by the joint torque limits. Once `|tau_ext|` on a joint approaches that limit there is
  no admissible `tau` left and QLD returns `fail code = 149` ("constraints inconsistent",
  constraint 149-100); mc_rtc logs `QP failed to run()` and emits *no* command, so mc_mujoco
  integrates garbage and reports `Nan, Inf or huge value in QACC at DOF n`. The wrist is where it
  bites: joints 5-7 are limited to **9 N.m** against 39 N.m for joints 1-4, so ~30 N on the plate
  at ~0.3 m already outruns joint_5. Mapping `QACC at DOF n` back to a joint needs the scene's
  worldbody order, not the arm's: with the `ground` robot and the stray `ball` from
  `~/.config/mc_rtc/mc_mujoco/mc_mujoco.yaml` present, `ball_ball`'s freejoint takes DOFs 0-5, the
  ground plane takes none, and the arm starts at DOF 6 -- so `DOF 10` is joint_5.
  `limitExternalTorques()` (called from `run()`) saturates the estimate at
  `max_external_torque_ratio` (0.7) of each joint's limit before the solver sees it, which keeps a
  feasible `tau` available. It is applied to `robot()` *and* `realRobot()`, and deliberately
  *after* `externalWrench_` is computed, so the saturation never rewrites what the dataset
  records. `collab_tau_ext_clamped_dofs` says when it is active.
  The same infeasibility has a second, independent route once the end-effector is freed: with
  `posture_compliant_stiffness: 0` and the 6d task's gains ramped to ~0, the arm's nullspace is
  almost unopposed, and an external force can wind up an internal motion (joints 1 and 3
  counter-rotating) that neither task observes until it hits the **joint velocity** bounds. That
  one is documented in `plugins/SimHumanPartner/SimHumanPartner.h`; the cure there was anchoring
  the partner's impedance where its hand is, and `posture_compliant_damping` is the only blunt
  knob against it.

- Build/dev-loop details (nix, direnv, the fast `.superbuild` loop vs `direnv reload`, `git add -N`
  for new files) are in `DEVELOPMENT.md` — do not duplicate them here.
