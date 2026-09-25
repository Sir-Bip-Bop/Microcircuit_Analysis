#!/usr/bin/env python3
"""
packet_sim.py

Motivation
----------
Every measure on the poster so far is computed on ongoing activity: how ordered
it is, how much each population predicts its own future, how much one population
predicts another. None of them answers the question the framing promises, which
is what happens to a signal entering the column. That requires putting a signal
in and following it.

This injects a brief excitatory packet into L4E and records the response of all
eight populations, repeating it many times at several amplitudes so that the
mutual information between what was injected and what each population did can be
estimated. Attenuation of that information from layer to layer is the
transmission measurement.

The stimulus is a direct Poisson pulse onto L4E rather than the thalamic
population of the Potjans-Diesmann model. That is deliberately the less
biological of the two options: the thalamic route targets L4 and L6 together
with anatomical probabilities, which makes the entry point ambiguous and the
resulting attenuation harder to attribute. Injecting into one population only
means any information found elsewhere arrived through the cortical circuitry,
which is the point of the measurement. The cost is that the result describes the
model's internal propagation rather than a physiological input pathway, and the
caption should say so.

Design notes that matter for the estimate:

  spacing     Presentations are spaced well beyond the autocorrelation time of
              the ongoing activity, which persists for several hundred ms in
              this regime. Spacing them tighter would let one trial's response
              contaminate the next, inflating the apparent information.
  jitter      Inter-stimulus intervals are jittered so the network cannot
              entrain to the presentation rhythm, which would produce
              information about stimulus timing rather than amplitude.
  order       Amplitudes are presented in random order, so slow drift in the
              network state cannot masquerade as an amplitude effect.
  catch       Amplitude zero is included as a level. Without it the measurement
              cannot distinguish "the response encodes how big the packet was"
              from "the response encodes that something happened".

What it measures
----------------
Nothing by itself. It writes, per drive:

    spikes_<pop>.dat    spike times from a recorder on each population
    schedule.json       onset time and amplitude level of every presentation
    meta.json           amplitudes, populations, simulation parameters

packet_mi.py turns those into the information measurement.

IMPORTANT: this script has not been run against NEST by its author. The
integration points with your microcircuit code are marked INTEGRATION below and
may need adapting to your module layout. Run it once with --dry-run first, which
builds and prints the schedule without importing NEST, then once with a single
drive and a short --n-trials to confirm it produces sane output before launching
the full set.

Usage
-----
    # 1. check the schedule and the expected cost, no NEST needed
    python packet_sim.py --dry-run --drives 09.07 12.12 16.19 \\
        --n-trials 150 --amplitudes 0 0.1 0.2 0.4

    # 2. one short run to confirm the integration works
    python packet_sim.py --drives 12.12 --n-trials 5 --out ./packet_out

    # 3. the full set
    python packet_sim.py --drives 09.07 12.12 16.19 \\
        --n-trials 150 --amplitudes 0 0.1 0.2 0.4 \\
        --isi-ms 600 --jitter-ms 100 --pulse-ms 10 \\
        --out ./packet_out
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import nest

POPS = ["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]


def snap(t, res_ms):
    """
    Move times onto the simulation resolution grid.

    NEST rejects any rate-change time for an inhomogeneous_poisson_generator
    that is not representable at the current resolution, and jittered onsets are
    arbitrary floats. Snapping is done by computing an integer number of
    resolution steps rather than by rounding in milliseconds, so no float dust
    survives to be rejected.
    """
    n = np.rint(np.asarray(t, dtype=np.float64) / res_ms).astype(np.int64)
    return n * res_ms


def build_schedule(n_trials, amplitudes, isi_ms, jitter_ms, warmup_ms, seed,
                   res_ms=0.1):
    """
    Randomised presentation schedule.

    Returns onset times in ms and the amplitude index of each presentation.
    Amplitudes are shuffled as a balanced block so every level appears equally
    often, rather than drawn independently, which with a few hundred trials
    would leave the levels noticeably unbalanced and bias the information
    estimate.
    """
    rng = np.random.default_rng(seed)
    n_lev = len(amplitudes)
    levels = np.tile(np.arange(n_lev), n_trials)
    rng.shuffle(levels)

    gaps = isi_ms + rng.uniform(-jitter_ms, jitter_ms, size=levels.size)
    onsets = snap(warmup_ms + np.cumsum(gaps) - gaps[0], res_ms)
    if np.any(np.diff(onsets) <= 0):
        raise ValueError("non-increasing onsets after snapping; reduce "
                         "--jitter-ms or raise --isi-ms")
    return onsets, levels


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drives", nargs="+", required=True,
                    help="background rates in spikes/s, as directory names or "
                         "plain numbers")
    ap.add_argument("--n-trials", type=int, default=150,
                    help="presentations PER AMPLITUDE LEVEL")
    ap.add_argument("--amplitudes", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.4],
                    help="pulse strength as a fraction of the standing external "
                         "drive to L4E. 0 must be included as a catch level.")
    ap.add_argument("--pulse-ms", type=float, default=10.0)
    ap.add_argument("--isi-ms", type=float, default=600.0,
                    help="mean inter-stimulus interval. Keep it well beyond the "
                         "autocorrelation time of the ongoing activity, which "
                         "extends several hundred ms here.")
    ap.add_argument("--jitter-ms", type=float, default=100.0)
    ap.add_argument("--warmup-ms", type=float, default=1000.0)
    ap.add_argument("--target-pop", type=int, default=2,
                    help="index of the injected population, 2 = L4E")
    ap.add_argument("--resolution-ms", type=float, default=0.1,
                    help="simulation resolution. Checked against the value in "
                         "sim_dict once NEST is loaded, and the schedule is "
                         "rebuilt if they disagree.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="./packet_out")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and report the schedule without importing NEST")
    args = ap.parse_args(argv)

    if 0.0 not in args.amplitudes:
        print("WARNING: no zero amplitude. Without a catch level the estimate "
              "cannot separate encoding of packet size from encoding of packet "
              "presence.")

    n_pres = args.n_trials * len(args.amplitudes)
    onsets, levels = build_schedule(args.n_trials, args.amplitudes,
                                    args.isi_ms, args.jitter_ms,
                                    args.warmup_ms, args.seed,
                                    args.resolution_ms)
    t_stop = float(snap(onsets[-1] + args.isi_ms, args.resolution_ms))

    print(f"presentations per drive : {n_pres} "
          f"({args.n_trials} x {len(args.amplitudes)} levels)")
    print(f"simulated time per drive: {t_stop / 1000:.1f} s")
    print(f"drives                  : {len(args.drives)}")
    print(f"total simulated time    : {len(args.drives) * t_stop / 1000:.1f} s")
    print(f"amplitudes (fraction of standing external drive): {args.amplitudes}")

    if args.dry_run:
        print("\ndry run, stopping before NEST import")
        return 0

    # ---------------- INTEGRATION ----------------
    # These imports and the Network construction follow the NEST microcircuit
    # example. Adapt to your module layout if it differs.
    import nest
    from network import Network
    from sim_params import sim_dict
    from network_params import net_dict
    from stimulus_params import stim_dict

    res = float(sim_dict.get("sim_resolution", args.resolution_ms))
    if abs(res - args.resolution_ms) > 1e-12:
        print(f"[sim] sim_dict resolution is {res} ms, not "
              f"{args.resolution_ms}; rebuilding the schedule on that grid")
        onsets, levels = build_schedule(args.n_trials, args.amplitudes,
                                        args.isi_ms, args.jitter_ms,
                                        args.warmup_ms, args.seed, res)
        t_stop = float(snap(onsets[-1] + args.isi_ms, res))
    if args.pulse_ms < res:
        raise SystemExit(f"--pulse-ms {args.pulse_ms} is shorter than the "
                         f"resolution {res}")

    os.makedirs(args.out, exist_ok=True)

    for drive in args.drives:
        bg = float(drive) if not any(c.isalpha() for c in str(drive)) \
            else float(drive)
        d_out = os.path.join(args.out, str(drive))
        os.makedirs(d_out, exist_ok=True)
        if os.path.exists(os.path.join(d_out, "schedule.json")):
            print(f"[sim] {drive}: already present, skipping")
            continue

        sd = dict(sim_dict)
        sd["data_path"] = d_out
        sd["t_sim"] = t_stop
        nd = dict(net_dict)
        nd["bg_rate"] = bg
        st = dict(stim_dict)
        st["thalamic_input"] = False
        st["dc_input"] = False

        # The Network class creates its own spike recorders, which duplicates
        # every spike on disk: at 361 s per drive that is several GB of
        # redundant output. This script records what it needs itself, so the
        # built-in devices are turned off where the key is recognised.
        for key in ("rec_dev", "recording_devices"):
            if key in sd:
                sd[key] = []
                print(f"[sim] disabled built-in recording via sim_dict['{key}']")
                break
        else:
            print("[sim] NOTE: could not find the recording-device key in "
                  "sim_dict, so spikes will be written twice. Harmless but it "
                  "doubles the disk usage.")
        nest.ResetKernel()
        net = Network(sd, nd, st)
        net.create()
        net.connect()

        # INTEGRATION: the example exposes the populations as net.pops and the
        # external synapse weight as net.weight_ext. If your version names them
        # differently this is the line to change.
        pops = getattr(net, "pops", None)
        if pops is None:
            raise SystemExit(
                "Network has no attribute 'pops'. Adapt the INTEGRATION block "
                "to expose the eight NodeCollections in creation order.")
        w_ext = getattr(net, "weight_ext", None)
        if w_ext is None:
            raise SystemExit(
                "Network has no attribute 'weight_ext'. Supply the external "
                "synaptic weight used for the background Poisson input.")

        target = pops[args.target_pop]
        k_ext = nd["K_ext"][args.target_pop]
        base_rate = k_ext * bg          # standing external input, spikes/s

        # One inhomogeneous generator encodes the whole schedule: rate steps up
        # at each onset and back to zero at the offset.
        offsets = snap(onsets + args.pulse_ms, res)
        times, values = [], []
        for t_on, t_off, lev in zip(onsets, offsets, levels):
            times.append(float(t_on))
            values.append(float(args.amplitudes[lev] * base_rate))
            times.append(float(t_off))
            values.append(0.0)
        tarr = np.asarray(times)
        if np.any(np.diff(tarr) <= 0):
            raise SystemExit("rate_times are not strictly increasing; the "
                             "pulse is as long as the shortest gap")
        gen = nest.Create("inhomogeneous_poisson_generator",
                          params={"rate_times": times, "rate_values": values})
        nest.Connect(gen, target,
                     syn_spec={"weight": w_ext, "delay": nd["delay_exc_mean"]})

        recs = []
        for p, pop in enumerate(pops):
            r = nest.Create("spike_recorder", params={
                "record_to": "ascii",
                "label": os.path.join(d_out, f"spikes_{POPS[p]}")})
            nest.Connect(pop, r)
            recs.append(r)

        nest.Simulate(t_stop)

        json.dump({"onsets_ms": [float(t) for t in onsets],
                   "levels": [int(l) for l in levels],
                   "amplitudes": list(args.amplitudes),
                   "pulse_ms": args.pulse_ms,
                   "target_pop": POPS[args.target_pop],
                   "bg_rate": bg,
                   "base_rate": base_rate,
                   "t_stop": t_stop,
                   "pops": POPS,
                   "n_full": [int(n) for n in nd["full_num_neurons"]]},
                  open(os.path.join(d_out, "schedule.json"), "w"), indent=2)
        print(f"[sim] {drive}: done, {n_pres} presentations, "
              f"{t_stop / 1000:.1f} s simulated")

    print("[sim] all drives complete")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)