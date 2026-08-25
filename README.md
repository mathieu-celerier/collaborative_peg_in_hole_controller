# CollabPegInHoleController

An mc_rtc controller for **collaborative peg-in-hole insertion by two people on one Kinova arm**,
used to record demonstrations for imitation learning.

- **Person A — the demonstrator** hand-guides the robot *body*, upstream of the F/T sensor. They
  are only visible in the joint-torque residual (`ExternalForcesObserver`).
- **Person B — the collaborator** manipulates the *piece* held at the end effector, downstream of
  the sensor. They are visible in the F/T wrench (`EEForceSensor`).

Both channels are logged separately so a policy can later be trained to take over person A's role.

## Build and run

Everything about the nix/direnv setup is in [`DEVELOPMENT.md`](DEVELOPMENT.md). In short:

```sh
direnv allow                                     # first time only
cmake -B build $cmakeFlags -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR
cmake --build build --target install             # fast loop, ~20 s
mc_mujoco                                        # or mc_kortex on the real robot
ctest --test-dir build --output-on-failure       # ticks the controller and drives 3 episodes
```

## Recording a demonstration

In the mc_rtc GUI:

1. `{Collab, Mode}` → **Go to Compliant** (the robot becomes hand-guidable; `Initial` is the stiff,
   position-controlled homing mode).
2. `{Collab, Episode}` → **Start episode**.
3. Perform the insertion together.
4. **Mark success** or **Mark failure** — the outcome is sticky and covers the whole episode, so it
   can be pressed any time before or just after stopping.
5. **Stop episode**.

Repeat. Everything lands in the ordinary mc_rtc session log
(`/tmp/mc-control-CollabPegInHoleController-<date>.bin`, symlinked as `...-latest.bin`), with the
episode boundaries marked inside it. The GUI shows the live episode id, recording flag, outcome and
duration; misuse is caught and warned about rather than silently corrupting a recording (a double
"Start episode" closes the running one, "Stop"/"Mark" with nothing recording is ignored).

### Checklist before a session

- `ExternalForcesObserver` is in the observer pipeline **and active** — without it there is no
  person-A channel and no force feedback at all. The controller warns once at startup if it is
  missing, and again if it is present but inactive.
- **The F/T calibration matches the interface.** Nothing to do — the controller detects
  mc_mujoco at reset and swaps in mc_kinova's simulation calibration, logging that it did.
  It matters because the robot's calibration carries the sensor's force gain and its home-pose
  tare, neither of which exists in MuJoCo: used there it leaves the plate's own 6.5 N in the
  wrench, the observer projects that in as a phantom external force, and entering Compliant
  kicks the arm with it. Confirm with `collab_ft_wrench_norm_nograv` ≈ 0 at rest — and check it
  at more than one pose, since the wrong calibration happens to read ~0 at the home pose.
- The F/T sensor reads non-zero (`{Collab, Compliance}` → "F/T wrench norm"). A sustained exact
  zero for >500 ticks is warned about: on the real robot it means the Bota topic is not publishing.
- The robot key in `etc/CollabPegInHoleController.yaml` is `kinova_bota_peg_plate`. Any other key
  and the whole tuning block is **silently ignored**.
- Thresholds in `{Collab, Compliance}` are sane for the piece in use — they are editable live and
  are logged per tick, so a mid-session change is recoverable from the data.

## Extracting a dataset

```sh
# what is in this log?
scripts/extract_episodes.py /tmp/mc-control-CollabPegInHoleController-latest.bin --list

# write one .npz per successful demonstration, resampled to 30 Hz
scripts/extract_episodes.py '/tmp/mc-control-*.bin' -o dataset/ --rate 30 --only-success
```

Options: `--rate` (target policy rate, 0 keeps the raw 1 kHz), `--only-success`, `--min-samples`,
`--list`/`--dry-run`. Force and residual channels are low-pass filtered before decimation — plain
striding from 1 kHz would alias sensor noise into the band the policy learns from.

Each output holds `t`, `obs/*`, `action/q_cmd` (+ `action/dq_cmd` deltas) and metadata (episode id,
outcome, dt, frames, source log). Requires the mc_rtc environment on `PATH` (it reads `.bin`
through `mc_log_ui`, falling back to `mc_bin_to_log`) plus numpy.

## What is logged (every control tick, 1 kHz)

| Entry | Channel |
|---|---|
| `collab_residual`, `collab_tau_ext`, `collab_tau_comp`, `collab_alphaD_ext` | **person A** — joint-torque residual |
| `collab_ft_wrench`, `collab_ft_wrench_world`, `collab_ft_wrench_norm_nograv`, `collab_tau_ft` | **person B** — F/T sensor |
| `collab_ext_wrench`, `collab_ext_wrench_norm` | **both**, reconstructed at the tool frame |
| `collab_residual_wrench`, `collab_residual_wrench_norm` | **person A** at the tool frame, same units as the row above |
| `collab_tool_pose`, `collab_insertion_pose` | task space |
| `collab_ee_free`, `collab_ee_stiffness`, `collab_trigger_norm*` | dual-compliance state |
| `collab_episode_id`, `collab_recording`, `collab_episode_outcome`, `collab_episode_tick` | episode segmentation |
| `collab_meta_*` | frames, sensor, trigger source and live tuning (provenance) |
| `qIn`, `alphaIn`, `tauIn`, `qOut`, `tauOut` | mc_rtc built-ins — robot state and commands |

`collab_residual` changed meaning with the move to `ExternalForcesObserver`: the observer feeds
the F/T torques into its momentum integral as a *known input*, so the residual it reports is the
part the sensor does not explain — person B is already subtracted. Under the old
`ExternalForcesEstimator` plugin the same entry carried the raw residual, which contained both
people. Logs recorded before the migration are not comparable on that entry; `collab_ext_wrench`
is unchanged. `collab_speed_residual` is gone — the observer has no speed-residual observer.

`tauIn` carries the measured joint torques — mc_mujoco and mc_kortex both feed them through
`setJointTorques`, and so does the test harness. If it is ever empty in a log, the interface is
not publishing torques, and `ExternalForcesObserver` has no input either if its
`torque_source_type` is set to a measurement rather than `CommandedTorque`.

## Layout

```
src/CollabPegInHoleController.{h,cpp}   controller: tasks, dual compliance, GUI, logging, episodes
src/states/                             Initial (stiff) and Compliant (hand-guidable) FSM states
etc/CollabPegInHoleController.in.yaml   FSM + per-robot tuning (heavily commented)
etc/CollabPegInHoleController/plugins/  per-controller plugin overrides
scripts/extract_episodes.py             log -> per-episode .npz dataset
tests/main.cpp                          headless run driving three recorded episodes
```

[mc_rtc]: https://jrl-umi3218.github.io/mc_rtc/
