#!/usr/bin/env python3
"""Extract collaborative peg-in-hole demonstrations from mc_rtc binary logs.

One "episode" is one demonstration, delimited in the log by the ``collab_recording`` flag that
the controller's GUI raises on "Start episode" and drops on "Stop episode". This script slices
the log on those edges, decimates from the 1 kHz control rate down to a policy rate, and writes
one ``.npz`` per episode ready for imitation-learning training.

Typical use::

    # what is in this log?
    ./extract_episodes.py /tmp/collab_episode_0003.bin --list

    # extract a whole session at the policy rate
    ./extract_episodes.py /tmp/collab_episode_*.bin -o dataset/ --rate 100

Pick the rate for the physics, not for the disk. `--rate` defaults to 30 Hz, which is a
reasonable policy rate and a poor archive rate: contact transients and the negotiation
between two partners live above it, and decimation cannot be undone once the binary log is
gone -- and the binary log cannot be kept, at ~30 MB/s of run time. The learning repo's
batch runner therefore extracts at 100 Hz.

Each output holds, per timestep:
  t                       time (s), rebased to the start of the episode
  obs/*                   observations (see OBSERVATIONS): robot state, person A's residual
                          channel, person B's F/T channel, tool and insertion poses
  action/q_cmd            commanded joint positions (qOut) -- the demonstrated action
  action/dq_cmd           qOut[k+1] - qOut[k], the same action as a delta
  target/<partner>_reference  where each partner's plan was asking that hand to be, world
                          frame -- the intent, and the second candidate action for a policy
                          (see PARTNER_REFERENCE_ENTRY)
  target/<partner>_hand   where that partner's hand actually was, world frame. With the
                          reference above, the spring the partner's force came out of
                          (see PARTNER_HAND_ENTRY)
  target/<partner>_force  the world-frame force each simulated partner actually applied
                          (see PARTNER_FORCE_ENTRY) -- the label for a policy that replaces
                          that partner, and never an observation
plus scalar metadata (episode id, outcome, dt, frames, source log) and, when the simulated
partners were randomised, ``meta/noise/*`` -- the draw that produced this episode. That is
metadata rather than an observation on purpose: see NOISE_ENTRY.

``meta/seated_depth`` is the graded outcome: how far the peg actually went in, recorded
whether or not the episode was scored a success. ``outcome`` is a threshold on it
(``meta/success_depth`` / ``meta/success_radius``, both recorded here for that reason), so a
batch that saturates can be re-scored rather than re-run.
"""

import argparse
import csv
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

# Log entries pulled into obs/, as {output name: log entry base name}. Missing entries are
# skipped with a warning rather than failing the whole extraction: a log recorded without the
# ExternalForcesObserver still has a usable F/T channel, and vice versa.
OBSERVATIONS = {
    # robot state (mc_rtc built-ins)
    "q": "qIn",
    "dq": "alphaIn",
    "tau": "tauIn",
    # person A: the demonstrator moving the arm, seen only through the joint-torque residual.
    # Since the move to ExternalForcesObserver the residual is the part the F/T sensor does
    # NOT explain, so person B is already subtracted from it; logs from before that carry the
    # raw residual under the same name and are not comparable on this key.
    "residual": "collab_residual",
    "tau_ext": "collab_tau_ext",
    # person B: the collaborator moving the piece, seen through the F/T sensor
    "ft_wrench": "collab_ft_wrench",
    "ft_wrench_world": "collab_ft_wrench_world",
    "tau_ft": "collab_tau_ft",
    # reconstructed at the tool frame: both people, then person A alone
    "ext_wrench": "collab_ext_wrench",
    "residual_wrench": "collab_residual_wrench",
    # the controller's own causal estimate of the partner it is working with -- six values,
    # the same ones cpih.policy.context.partner_context computes at export time. Carried so
    # that check_context_agreement.py can compare the two: they are supposed to be identical,
    # and nothing else would notice if they stopped being.
    "partner_context": "collab_partner_context",
    # task space
    "tool_pose": "collab_tool_pose",
    "insertion_pose": "collab_insertion_pose",
}

ACTION_ENTRY = "qOut"

# Channels that are decimated with an anti-aliasing filter rather than by plain striding.
# Force/torque channels carry structural noise well above the policy rate; naive subsampling
# folds it back into the band the policy learns from.
SMOOTH_PREFIXES = ("residual", "tau", "ft_wrench", "ext_wrench", "dq")

# The per-episode noise draw, carried through as METADATA rather than as an observation.
#
# Two reasons it is not an obs/ key. It is constant across an episode, so as a per-timestep
# channel it is a column of the same number repeated; and it is privileged information --
# a partner's private belief about where the peg is, and how hard they decided to push today,
# are exactly the things the dyad has to infer from each other's forces. Handing them to a
# policy is the same mistake as handing it person A's residual.
#
# The partner names are not known here (they come from the generated config), so the entries
# are discovered from the log rather than listed. Without them an episode cannot be
# reproduced or conditioned on: the generated yaml holds the distribution, not the sample.
NOISE_ENTRY = re.compile(
    r"^sim_human_(noise_seed|noise_registration|.+_(target_offset|gain_scale|delay))$"
)

# The force each simulated partner actually applied, world frame, one entry of three columns
# per partner. This is the *label* for a policy trained to replace that partner, and it is
# deliberately not an obs/ key.
#
# It is not the same quantity as obs/residual_wrench, and the difference decides which one a
# trainer should use. residual_wrench is the momentum observer's *estimate* of person A,
# reconstructed at the tool frame and carrying the observer's own dynamics; this is the
# ground truth the plugin handed to mc_mujoco, in the frame and units the plugin will emit
# again at test time. A policy that replaces the partner's spring-damper has to reproduce
# this one -- training on the estimate makes the policy chase a signal that is itself a
# filtered function of what it outputs.
#
# Only available in simulation, which is the only place demonstrations currently come from.
# A log without the plugin simply has no such entries and the key is absent.
PARTNER_FORCE_ENTRY = re.compile(r"^sim_human_(.+)_force$")

# The other half of a partner: where its minimum-jerk plan was asking that hand to be, world
# frame. The force is what the impedance produced from the gap between this and the hand's
# actual position, so the two together are the whole partner model.
#
# Carried because it is a second candidate action for a policy. A policy that predicts the
# force is a reactive map -- it can follow, but it has nothing to generate a motion from, and
# measured on the first evaluation it led at 5 % where it followed at 49 %. A policy that
# predicts the *intent* emits a trajectory instead, and the existing impedance turns that back
# into a force. It is also the action space RoboManipBaselines takes natively
# (COMMAND_EEF_POSE), where a wrench action needs its DataKey list patched.
PARTNER_REFERENCE_ENTRY = re.compile(r"^sim_human_(.+)_reference$")

# Where that partner's hand actually was, world frame -- the point the plugin measured its
# reference against, `aimedPoint(referenceBody, referencePoint)`.
#
# Carried because the *difference* `reference - hand` is a third candidate action, and the
# one the impedance actually consumes: the force is `K (reference - hand) + D (...)`, so a
# policy predicting the difference predicts a bounded force by construction, where one
# predicting the absolute reference can ask for any stretch at all. Deriving it afterwards
# from `obs/tool_pose` needs a constant body-frame offset that is nowhere in the `.npz`, so
# it is written down instead of reconstructed.
PARTNER_HAND_ENTRY = re.compile(r"^sim_human_(.+)_hand$")

# Metadata that is a vector rather than a scalar, so it needs `gather` rather than
# `scalar_metadata`. Constant across an episode, hence metadata and not an observation.
VECTOR_METADATA = {
    # Where the peg is, world frame. Scene knowledge the controller never has, and the
    # anchor every placement-independent feature is built against: the error a partner is
    # closing is (hole - peg), and obs/insertion_pose only gives the hole. Without it an
    # .npz is not self-contained and episodes from different placements cannot be pooled.
    "peg_target": "sim_human_target",
}

METADATA_ENTRIES = {
    "tool_frame": "collab_meta_tool_frame",
    "ft_sensor": "collab_meta_ft_sensor",
    "insertion_frame": "collab_meta_insertion_frame",
    "trigger_source": "collab_meta_trigger_source",
    "residual_gain": "collab_meta_residual_gain",
    "dual_force_enter": "collab_meta_dual_force_enter",
    "dual_force_exit": "collab_meta_dual_force_exit",
    # The graded outcome, and the thresholds it was scored against. Read at the episode's
    # last tick like the rest of this table, which is correct because the plugin holds them
    # flat once set: `seated_depth` is a running maximum and `time_to_seat` is written once.
    #
    # These are what make `outcome` re-derivable. An episode run with `stop_on_success:
    # false` records how deep the dyad actually got, so a batch can be re-scored at a
    # different depth without re-simulating -- and a dataset whose success rate sits at 97 %
    # can be looked at again rather than thrown away.
    "seated_depth": "sim_human_seated_depth",
    "seated_lateral": "sim_human_seated_lateral",
    "time_to_seat": "sim_human_time_to_seat",
    "success_depth": "sim_human_success_depth",
    "success_radius": "sim_human_success_radius",
}


# --- log reading ----------------------------------------------------------------------------


def read_log(path):
    """Read an mc_rtc log into {entry name: 1-D array}, preserving log column order.

    Prefers mc_log_ui (numpy, fast); falls back to `mc_bin_to_log` + csv so the script keeps
    working in a shell where mc_log_ui's Qt dependencies are unavailable.
    """
    try:
        from mc_log_ui import read_log as mc_read_log

        return mc_read_log(path)
    except ImportError:
        pass
    if not path.endswith(".bin"):
        raise SystemExit(f"{path}: mc_log_ui unavailable, only .bin logs can be read")
    if shutil.which("mc_bin_to_log") is None:
        raise SystemExit(
            "neither mc_log_ui nor mc_bin_to_log is available - source the mc_rtc environment"
        )
    tmpdir = tempfile.mkdtemp(prefix="collab_extract_")
    try:
        # mc_bin_to_log writes exactly the name it is given, without appending an extension,
        # so ask for the .csv outright rather than guessing it back afterwards.
        out = os.path.join(tmpdir, "log.csv")
        subprocess.run(["mc_bin_to_log", path, out], check=True)
        return _read_csv(out)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _read_csv(path):
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        columns = [[] for _ in header]
        for row in reader:
            for i, value in enumerate(row):
                columns[i].append(value)
    data = {}
    for name, values in zip(header, columns):
        try:
            data[name] = np.array(values, dtype=float)
        except ValueError:
            data[name] = np.array(values, dtype=object)
    return data


# Component suffixes mc_rtc appends when it flattens a spatial type: ForceVecd/MotionVecd
# (couple/force, angular/linear) and PTransformd (quaternion + translation). Vectors are
# flattened as plain indices instead.
COMPONENT_SUFFIXES = {
    "cx", "cy", "cz", "fx", "fy", "fz",
    "wx", "wy", "wz", "vx", "vy", "vz",
    "qw", "qx", "qy", "qz", "tx", "ty", "tz",
    "x", "y", "z", "w",
}


def gather(data, base):
    """Collect the flattened columns of one log entry into an (N, D) array + column names.

    Only exact component suffixes are accepted: `collab_ft_wrench` and `collab_ft_wrench_world`
    are distinct entries sharing a prefix, so a plain startswith() would splice one into the
    other. Numeric suffixes sort numerically (_10 after _9); component suffixes keep log order,
    which is the order the logger wrote them in.
    """
    if base in data:
        return np.asarray(data[base]).reshape(-1, 1), [base]
    keys = [
        k
        for k in data
        if k.startswith(base + "_")
        and (k[len(base) + 1 :].isdigit() or k[len(base) + 1 :] in COMPONENT_SUFFIXES)
    ]
    if not keys:
        return None, []
    if all(k[len(base) + 1 :].isdigit() for k in keys):
        keys.sort(key=lambda k: int(k[len(base) + 1 :]))
    return np.column_stack([np.asarray(data[k], dtype=float) for k in keys]), keys


def noise_entries(data):
    """The noise log entries present, as base names, in log order.

    A base name is what `gather` takes: `sim_human_person_a_demonstrator_target_offset` is
    one entry with three columns, and its columns must not be mistaken for three entries.
    """
    bases = []
    for key in data:
        if not key.startswith("sim_human_"):
            continue
        for candidate in (key, key.rsplit("_", 1)[0]):
            if NOISE_ENTRY.match(candidate) and candidate not in bases:
                bases.append(candidate)
                break
    return bases


def partner_force_entries(data):
    """The partners' applied-force entries present, as base names, in log order.

    Discovered rather than listed for the same reason the noise entries are: the partner
    names come from the generated config and are not known here.
    """
    bases = []
    for key in data:
        if not key.startswith("sim_human_"):
            continue
        base = key.rsplit("_", 1)[0]
        if PARTNER_FORCE_ENTRY.match(base) and base not in bases:
            bases.append(base)
    return bases


def partner_reference_entries(data):
    """The `sim_human_<partner>_reference` entries, discovered the same way."""
    bases = []
    for key in data:
        if not key.startswith("sim_human_"):
            continue
        base = key.rsplit("_", 1)[0]
        if PARTNER_REFERENCE_ENTRY.match(base) and base not in bases:
            bases.append(base)
    return bases


def partner_hand_entries(data):
    """The `sim_human_<partner>_hand` entries, discovered the same way."""
    bases = []
    for key in data:
        if not key.startswith("sim_human_"):
            continue
        base = key.rsplit("_", 1)[0]
        if PARTNER_HAND_ENTRY.match(base) and base not in bases:
            bases.append(base)
    return bases


def scalar_metadata(data, entry, index):
    """Read one metadata entry at a given sample (strings are logged per tick)."""
    if entry not in data:
        return None
    value = np.asarray(data[entry])[index]
    return value.item() if hasattr(value, "item") else value


# --- segmentation ---------------------------------------------------------------------------


class Episode:
    def __init__(self, source, start, stop, episode_id, outcome):
        self.source = source
        self.start = start  # first sample, inclusive
        self.stop = stop  # last sample, exclusive
        self.id = episode_id
        self.outcome = outcome

    def __len__(self):
        return self.stop - self.start


def segment(data, source, min_samples):
    """Find the recording spans of a log, on the rising/falling edges of collab_recording."""
    if "collab_recording" not in data:
        raise SystemExit(
            f"{source}: no 'collab_recording' entry - was this log produced by "
            "CollabPegInHoleController?"
        )
    recording = np.asarray(data["collab_recording"], dtype=float) > 0.5
    ids = np.asarray(data.get("collab_episode_id", np.zeros(len(recording))), dtype=float)
    outcomes = data.get("collab_episode_outcome")

    # An episode that was still recording when the log ended is closed at the last sample; the
    # padded edges below make that fall out of the same diff.
    edges = np.diff(np.concatenate(([False], recording, [False])).astype(np.int8))
    starts = list(np.flatnonzero(edges == 1))
    stops = list(np.flatnonzero(edges == -1))

    # An episode stopped and the next one started between two control ticks (a fast
    # stop-then-start, or "Start episode" pressed twice) leaves no sample with recording=0 in
    # between, so the recording edges alone would merge them. The episode id still changes, so
    # split on that too.
    spans = []
    for start, stop in zip(starts, stops):
        cuts = start + 1 + np.flatnonzero(np.diff(ids[start:stop]) != 0)
        for a, b in zip([start, *cuts], [*cuts, stop]):
            spans.append((a, b))

    episodes = []
    for start, stop in spans:
        episode_id = int(ids[stop - 1])
        outcome = "unknown"
        if outcomes is not None:
            # The outcome is sticky and only cleared by the next "Start episode", so it stays
            # valid past the end of the recording span. Search up to the next episode id to also
            # catch a "Mark success" pressed after "Stop episode" (or between two control ticks,
            # where no sample of the span itself carries it).
            end = stop + int(np.argmax(ids[stop:] != episode_id)) if stop < len(ids) else stop
            if end == stop and stop < len(ids) and ids[stop] == episode_id:
                end = len(ids)  # the id never changes again: valid to the end of the log
            values = [str(x) for x in np.asarray(outcomes)[start:max(end, stop)]]
            marked = [v for v in values if v != "none"]
            outcome = marked[-1] if marked else "none"
        elif "collab_episode_label" in data:
            # Logs from before collab_episode_outcome existed: the label was overwritten by
            # "idle" on stop, so the outcome is only recoverable from the last labelled sample.
            labels = np.asarray(data["collab_episode_label"])[start:stop]
            marked = [str(x) for x in labels if str(x) not in ("recording", "idle")]
            outcome = marked[-1] if marked else "unknown"
        if stop - start < min_samples:
            print(
                f"  - skipping episode {episode_id} in {os.path.basename(source)}: "
                f"{stop - start} samples is below --min-samples"
            )
            continue
        episodes.append(Episode(source, start, stop, episode_id, outcome))
    return episodes


# --- decimation -----------------------------------------------------------------------------


def moving_average(x, window):
    """Zero-phase box filter along axis 0, edges handled by reflection."""
    if window < 2:
        return x
    if window % 2 == 0:
        window += 1
    half = window // 2
    padded = np.pad(x, ((half, half), (0, 0)), mode="reflect")
    kernel = np.ones(window) / window
    return np.stack(
        [np.convolve(padded[:, c], kernel, mode="valid") for c in range(x.shape[1])], axis=1
    )


def decimate(x, factor, smooth):
    if factor <= 1:
        return x
    return (moving_average(x, factor) if smooth else x)[::factor]


# --- extraction -----------------------------------------------------------------------------


def extract(episode, data, args):
    sl = slice(episode.start, episode.stop)
    t, _ = gather(data, "t")
    if t is None:
        raise SystemExit(f"{episode.source}: no 't' entry in the log")
    t = t[sl]
    dt = float(np.median(np.diff(t[:, 0]))) if len(t) > 1 else 0.0

    factor = 1
    if args.rate and dt > 0:
        factor = max(1, int(round(1.0 / (args.rate * dt))))

    out = {"t": decimate(t - t[0], factor, False)[:, 0]}
    missing = []
    for name, entry in OBSERVATIONS.items():
        values, columns = gather(data, entry)
        if values is None:
            missing.append(entry)
            continue
        smooth = name.startswith(SMOOTH_PREFIXES)
        out["obs/" + name] = decimate(values[sl], factor, smooth)
        out["columns/" + name] = np.array(columns)

    action, columns = gather(data, ACTION_ENTRY)
    if action is None:
        missing.append(ACTION_ENTRY)
    else:
        q_cmd = decimate(action[sl], factor, False)
        out["action/q_cmd"] = q_cmd
        # Delta action, defined at the same timestamps as q_cmd; the last row repeats zero so
        # every array in the file has the same length.
        out["action/dq_cmd"] = np.vstack([np.diff(q_cmd, axis=0), np.zeros((1, q_cmd.shape[1]))])
        out["columns/q_cmd"] = np.array(columns)

    # Smoothed on the way down like the other force channels: this is a 1 kHz signal and
    # plain striding would fold its structure back into the band the policy learns from.
    for entry in partner_force_entries(data):
        values, columns = gather(data, entry)
        if values is None:
            continue
        name = entry[len("sim_human_") :]
        out["target/" + name] = decimate(values[sl], factor, True)
        out["columns/" + name] = np.array(columns)

    # Plain striding, unlike the forces: this is a minimum-jerk reference, already smooth by
    # construction, and averaging it would round the corners of the plan the policy is meant
    # to reproduce.
    for entry in partner_reference_entries(data):
        values, columns = gather(data, entry)
        if values is None:
            continue
        name = entry[len("sim_human_") :]
        out["target/" + name] = decimate(values[sl], factor, False)
        out["columns/" + name] = np.array(columns)

    # Struck rather than averaged for the same reason as the reference: the difference of the
    # two is the quantity of interest, and averaging one of a pair while striding the other
    # would put a filter's lag into their difference.
    for entry in partner_hand_entries(data):
        values, columns = gather(data, entry)
        if values is None:
            continue
        name = entry[len("sim_human_") :]
        out["target/" + name] = decimate(values[sl], factor, False)
        out["columns/" + name] = np.array(columns)

    out["episode_id"] = episode.id
    out["outcome"] = episode.outcome
    out["dt"] = dt * factor
    out["source_log"] = episode.source
    out["source_range"] = np.array([episode.start, episode.stop])
    for name, entry in METADATA_ENTRIES.items():
        value = scalar_metadata(data, entry, episode.stop - 1)
        if value is not None:
            out["meta/" + name] = value
    for name, entry in VECTOR_METADATA.items():
        values, _ = gather(data, entry)
        if values is not None and len(values):
            out["meta/" + name] = values[min(max(episode.start, 0), len(values) - 1)]
    for entry in noise_entries(data):
        values, _ = gather(data, entry)
        if values is None:
            continue
        # Read one sample from inside the episode rather than at its edge: the draw is made
        # when the episode opens, and the samples on either side of a boundary belong to a
        # different draw.
        row = values[min(max(episode.start, 0), len(values) - 1)]
        name = entry[len("sim_human_") :]
        out["meta/noise/" + name] = row[0] if row.size == 1 else row
    return out, missing


def summarize(episode, arrays):
    duration = float(arrays["t"][-1]) if len(arrays["t"]) else 0.0
    wrench = arrays.get("obs/ft_wrench")
    peak = float(np.linalg.norm(wrench, axis=1).max()) if wrench is not None and len(wrench) else 0.0
    return (
        f"  episode {episode.id:>4}  {duration:7.2f}s  {len(arrays['t']):>6} samples  "
        f"outcome={episode.outcome:<8} peak|F/T|={peak:7.2f}  ({os.path.basename(episode.source)})"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", help="mc_rtc .bin (or .flat/.csv) logs")
    parser.add_argument("-o", "--output", default="dataset", help="output directory (default: dataset)")
    parser.add_argument("--rate", type=float, default=30.0, help="target policy rate in Hz, 0 to keep 1 kHz")
    parser.add_argument("--only-success", action="store_true", help="keep only episodes marked 'success'")
    parser.add_argument("--min-samples", type=int, default=100, help="drop episodes shorter than this")
    parser.add_argument("--list", "--dry-run", dest="list_only", action="store_true",
                        help="print the segmentation without writing anything")
    args = parser.parse_args()

    paths = sorted({p for pattern in args.logs for p in (glob.glob(pattern) or [pattern])})
    if not args.list_only:
        os.makedirs(args.output, exist_ok=True)

    total, written = 0, 0
    for path in paths:
        if not os.path.exists(path):
            print(f"{path}: no such file", file=sys.stderr)
            continue
        print(f"{path}:")
        data = read_log(path)
        episodes = segment(data, path, args.min_samples)
        if not episodes:
            print("  (no recorded episode)")
            continue
        for episode in episodes:
            total += 1
            if args.only_success and episode.outcome != "success":
                print(f"  - episode {episode.id}: outcome '{episode.outcome}', skipped")
                continue
            arrays, missing = extract(episode, data, args)
            print(summarize(episode, arrays))
            if missing:
                print(f"      missing log entries: {', '.join(missing)}")
            if args.list_only:
                continue
            name = f"episode_{episode.id:04d}_{episode.outcome}.npz"
            np.savez_compressed(os.path.join(args.output, name), **arrays)
            written += 1

    print(f"\n{total} episode(s) found, {written} written to {args.output}"
          if not args.list_only else f"\n{total} episode(s) found")


if __name__ == "__main__":
    main()
