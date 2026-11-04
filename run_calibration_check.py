"""
run_calibration_check.py

The check the README asked for, done properly, at the REAL scale.

The README's "Calibration finding" says: don't trust any (eps_age,
plasticity_rate) pair -- including the ones already used in
run_production_sweep.py -- without first confirming monotonic convergence,
and calls this "a five-minute check" on "a couple of single realizations."

That check was never run before the production sweep, and based on what's
actually in production_results.jsonl, a couple of realizations would not
have been enough to catch the problem anyway: of the 102 completed
realizations (N=100, dZ=0.5, eps_age=0.04, plasticity_rate=0.012 -- the
"conservative" pair the README said was safe), 21 of them (21%) show error
dropping substantially and then REVERSING back up before cycle 800. A
failure rate that low has a good chance of looking clean across just one or
two spot checks.

This script instead runs enough realizations per (eps_age, plasticity_rate)
pair, AT THE SAME SCALE AS PRODUCTION (N=100, dZ=0.5, 800 cycles by
default), to actually estimate the reversal rate rather than guess at it
from a couple of lucky-or-unlucky realizations. It includes the exact pair
already used in production as one of the settings swept, specifically so
its own output can be checked against the 21/102 already found in
production_results.jsonl.

Run in the same folder as spring_network.py and training.py:
    python run_calibration_check.py
    python run_calibration_check.py --n-workers 8
    python run_calibration_check.py --n-realizations 40
    python run_calibration_check.py --cycles 400
    python run_calibration_check.py --quick       (3-realization smoke test,
                                                     small N, short budget --
                                                     confirms the environment
                                                     works, NOT real data)

Results append to calibration_results.jsonl as jobs complete, one JSON line
per realization, same pattern as the other sweep scripts -- safe to kill
and resume reading partial output at any time.
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from spring_network import SpringNetwork
from training import train_one_cycle, measure_error_and_order_parameter, lowest_modes_with_overlap

N = 100
DZ = 0.5
N_SUBSTEPS = 8
MEASURE_EVERY = 20
CYCLES = 800          # match run_production_sweep.py's budget by default
N_REALIZATIONS = 24   # per setting -- see "How many realizations" below

# A handful of (eps_age, plasticity_rate) pairs to compare, at the REAL
# scale (N=100), which the README's calibration note never actually tested
# -- it was only checked at the demo's N=35-50 scale. The first row is the
# exact pair run_production_sweep.py already used, included on purpose so
# this script's own output is checkable against the 21/102 reversal rate
# already found by hand in production_results.jsonl.
SETTINGS = [
    {"label": "production (already used)", "eps_age": 0.04, "plasticity_rate": 0.012},
    {"label": "README conservative",        "eps_age": 0.03, "plasticity_rate": 0.010},
    {"label": "more conservative still",     "eps_age": 0.02, "plasticity_rate": 0.008},
    {"label": "just past conservative",      "eps_age": 0.05, "plasticity_rate": 0.015},
]

# n_targets to cycle through within each setting, so the check isn't blind
# to the one axis (n_targets / delta) the production sweep actually varies.
N_TARGETS_GRID = [2, 5, 8, 12]

# Seed space kept well clear of every seed already used in
# production_results.jsonl / early_warning_results.jsonl (100000-1500005),
# so nothing here can ever collide with or be confused for that data.
SEED_BASE = 9_000_000

OUT_PATH = "calibration_results.jsonl"


# ----------------------------------------------------------------------
# The reversal diagnostic -- same definition used to find the 21/102 in
# production_results.jsonl: a real drop from the initial error (>0.1),
# followed by a climb-back from the minimum (>0.05), with the minimum not
# sitting at the very first or very last measurement.
# ----------------------------------------------------------------------
def classify_reversal(trajectory, drop_threshold=0.1, climb_threshold=0.05):
    errs = np.array([p["error"] for p in trajectory])
    cycles = np.array([p["cycle"] for p in trajectory])
    imin = int(np.argmin(errs))
    init_err, min_err, final_err = float(errs[0]), float(errs[imin]), float(errs[-1])
    real_drop = init_err - min_err
    climb_back = final_err - min_err
    reversed_ = (
        cycles[imin] not in (cycles[0], cycles[-1])
        and real_drop > drop_threshold
        and climb_back > climb_threshold
    )
    return {
        "reversed": bool(reversed_),
        "cycle_at_min": int(cycles[imin]),
        "init_error": init_err,
        "min_error": min_err,
        "final_error": final_err,
    }


def run_one(job):
    setting_idx, n_targets, rep, cfg = job
    seed = SEED_BASE + setting_idx * 1_000_000 + n_targets * 10_000 + rep
    rng = np.random.default_rng(seed)
    net = SpringNetwork.random_diluted_delaunay(N=cfg["N"], target_Z=4.0 + cfg["dZ"], rng=rng)
    nodes = rng.permutation(cfg["N"])
    source = tuple(nodes[:2])
    targets = [tuple(nodes[2 + 2 * t: 4 + 2 * t]) for t in range(n_targets)]
    phases = rng.choice([-1.0, 1.0], size=n_targets)

    traj = []
    t0 = time.time()
    for cycle in range(cfg["cycles"] + 1):
        if cycle % cfg["measure_every"] == 0:
            err, conv = measure_error_and_order_parameter(net, source, targets, phases)
            modes = lowest_modes_with_overlap(net, source, n_modes=6)
            traj.append({
                "cycle": cycle, "error": err, "converged": conv,
                "omega_trained": modes["omega_trained"],
                "omega_next": modes["omega_next"],
                "overlap": modes["overlap_trained"],
                "rank": modes["rank_of_trained"],
            })
        if cycle < cfg["cycles"]:
            train_one_cycle(net, source, targets, phases, eps_age=cfg["eps_age"],
                             plasticity_rate=cfg["plasticity_rate"], n_substeps=cfg["n_substeps"])

    diag = classify_reversal(traj)
    return {
        "setting_label": cfg["label"], "eps_age": cfg["eps_age"],
        "plasticity_rate": cfg["plasticity_rate"],
        "N": cfg["N"], "dZ": cfg["dZ"], "n_targets": n_targets,
        "seed": seed, "wall_time_s": time.time() - t0,
        **diag,
        "trajectory": traj,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=None,
                         help="default: all logical cores")
    parser.add_argument("--n-realizations", type=int, default=N_REALIZATIONS,
                         help=f"realizations per (eps_age, plasticity_rate) setting "
                              f"(default {N_REALIZATIONS} -- see script docstring for why "
                              f"'a couple' isn't enough to trust a ~20%%-level effect)")
    parser.add_argument("--cycles", type=int, default=CYCLES,
                         help=f"cycle budget per realization (default {CYCLES}, matching "
                              f"run_production_sweep.py)")
    parser.add_argument("--quick", action="store_true",
                         help="tiny smoke test (small N, 3 realizations/setting, short "
                              "budget) to confirm the environment works -- NOT real data")
    args = parser.parse_args()

    if args.quick:
        cfg_base = dict(N=40, dZ=DZ, cycles=60, n_substeps=N_SUBSTEPS, measure_every=20)
        n_reps = 3
        print("QUICK SMOKE TEST -- not real data, just checking it runs.")
    else:
        cfg_base = dict(N=N, dZ=DZ, cycles=args.cycles, n_substeps=N_SUBSTEPS,
                        measure_every=MEASURE_EVERY)
        n_reps = args.n_realizations

    # Config is baked into every job tuple and pickled to each worker
    # explicitly -- this is what actually reaches spawned processes on
    # Windows/macOS, unlike a module-global overwritten only in this
    # (parent) process, which a fresh re-imported worker never sees.
    jobs = []
    for setting_idx, setting in enumerate(SETTINGS):
        cfg = dict(cfg_base, label=setting["label"], eps_age=setting["eps_age"],
                   plasticity_rate=setting["plasticity_rate"])
        for rep in range(n_reps):
            n_targets = N_TARGETS_GRID[rep % len(N_TARGETS_GRID)]
            jobs.append((setting_idx, n_targets, rep, cfg))

    n_workers = args.n_workers or os.cpu_count()
    print(f"{len(jobs)} realizations queued across {n_workers} workers "
          f"({len(SETTINGS)} settings x {n_reps} realizations each, "
          f"N={cfg_base['N']}, dZ={cfg_base['dZ']}, cycles={cfg_base['cycles']})")

    t0 = time.time()
    done = 0
    by_setting = {s["label"]: [] for s in SETTINGS}
    # 'if __name__ == "__main__"' guard below is required on Windows for
    # multiprocessing to work at all -- don't move this call above it.
    with mp.Pool(n_workers) as pool, open(OUT_PATH, "a") as f:
        for result in pool.imap_unordered(run_one, jobs):
            f.write(json.dumps(result) + "\n")
            f.flush()
            done += 1
            by_setting[result["setting_label"]].append(result["reversed"])
            elapsed = time.time() - t0
            rate = done / elapsed
            eta_min = (len(jobs) - done) / rate / 60 if rate > 0 else float("nan")
            print(f"[{elapsed/60:6.1f} min] {done}/{len(jobs)} done "
                  f"({result['setting_label']}, n_targets={result['n_targets']}, "
                  f"reversed={result['reversed']}) -- ETA {eta_min:.0f} min", flush=True)

    print(f"\nAll done in {(time.time()-t0)/60:.1f} minutes. "
          f"Results in {OUT_PATH} ({done} realizations).\n")

    print(f"{'setting':>28} {'eps_age':>8} {'plast.':>7} {'n':>4} {'reversed':>9} {'rate':>6}")
    for setting in SETTINGS:
        label = setting["label"]
        vals = by_setting[label]
        if not vals:
            continue
        rate = sum(vals) / len(vals)
        print(f"{label:>28} {setting['eps_age']:8.3f} {setting['plasticity_rate']:7.3f} "
              f"{len(vals):4d} {sum(vals):9d} {rate:6.2f}")

    print("\nThe first row is the exact pair already used in run_production_sweep.py --\n"
          "compare its reversal rate here against the 21/102 (~0.21) already found by\n"
          "re-checking production_results.jsonl directly. If this script's rate for that\n"
          "row lands well outside that range, something about this run (seeds, machine,\n"
          "scipy/BLAS version) is different enough from the original production run that\n"
          "the comparison across settings below should be treated cautiously too.")


if __name__ == "__main__":
    main()