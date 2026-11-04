"""
run_second_dz_sweep.py

A scoped-down replication of run_production_sweep.py at a SECOND value of
dZ, to test whether the divergence phenomenon (Table 1) and its
non-monotonic dependence on target count are specific to dZ=0.5 or hold
more generally. This is the other likely referee ask: everything in the
manuscript is reported for a single coordination number.

This is NOT a full second Table 1 -- it's deliberately smaller (4
realizations per target count instead of 5-12) to keep the cost down
while still being informative: if the divergence rate and its
non-monotonic shape look qualitatively similar at a second dZ, that's
good evidence the phenomenon isn't an artifact of one corner of parameter
space, even without full statistical power to characterize it as
precisely as Table 1 does. If you want full power at the second dZ too,
raise --n-realizations to match (5-12, costing roughly what the original
production sweep did).

Default dZ is 1.0, chosen deliberately further from isostaticity than
0.5, since the manuscript already flags results near dZ=0 as only
qualitatively reliable given the Delaunay-dilute proxy -- testing a
second point that ALSO sits in that caveat's blind spot wouldn't add
much. Override with --dz if you want a different comparison point
instead (e.g. --dz 0.2 to specifically probe closer to isostaticity,
understanding that caveat applies there).

COST: 12 target-count values x 4 realizations = 48 jobs at N=100, 800
cycles -- roughly 48/102 of the original production sweep's cost (that
one took ~7 hours on your machine with 16 workers; expect in the
neighborhood of 3-3.5 hours for this one, same hardware).

Usage (same folder as spring_network.py and training.py):
    python run_second_dz_sweep.py
    python run_second_dz_sweep.py --dz 0.2
    python run_second_dz_sweep.py --n-realizations 6   # full power, costlier
    python run_second_dz_sweep.py --n-workers 8
    python run_second_dz_sweep.py --quick                # smoke test

Appends to second_dz_results.jsonl as jobs complete.
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
DZ_DEFAULT = 1.0
EPS_AGE = 0.04
PLASTICITY_RATE = 0.012
N_SUBSTEPS = 8
MEASURE_EVERY = 20
CYCLES = 800
N_REALIZATIONS_DEFAULT = 4

TARGET_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]  # same grid as Table 1

OUT_PATH = "second_dz_results.jsonl"

# Same reversal/divergence test used in run_calibration_check.py, for
# consistency when this data gets compared against the original sweep.
def classify_divergent(trajectory, peak_ratio_threshold=1.5):
    errs = np.array([p["error"] for p in trajectory])
    return bool(errs.max() / errs[0] > peak_ratio_threshold)


def run_one(job):
    n_targets, rep, dz, cfg = job
    # seed space kept well clear of the original dZ=0.5 sweep's seeds
    # (100000-1500011) and the calibration-check seeds (9000000+)
    seed = 20_000_000 + n_targets * 100_000 + rep
    rng = np.random.default_rng(seed)
    net = SpringNetwork.random_diluted_delaunay(N=cfg["N"], target_Z=4.0 + dz, rng=rng)
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
            traj.append({"cycle": cycle, "error": err, "converged": conv,
                         "omega_trained": modes["omega_trained"], "omega_next": modes["omega_next"],
                         "overlap": modes["overlap_trained"], "rank": modes["rank_of_trained"]})
        if cycle < cfg["cycles"]:
            train_one_cycle(net, source, targets, phases, eps_age=cfg["eps_age"],
                             plasticity_rate=cfg["plasticity_rate"], n_substeps=cfg["n_substeps"])

    return {
        "N": cfg["N"], "dZ": dz, "n_targets": n_targets, "seed": seed,
        "wall_time_s": time.time() - t0,
        "divergent": classify_divergent(traj),
        "final_error": traj[-1]["error"],
        "peak_error": max(p["error"] for p in traj),
        "trajectory": traj,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--dz", type=float, default=DZ_DEFAULT)
    parser.add_argument("--n-realizations", type=int, default=N_REALIZATIONS_DEFAULT,
                         help=f"realizations per target count (default {N_REALIZATIONS_DEFAULT}; "
                              f"use 5-12 to match Table 1's power, at roughly proportional cost)")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        cfg = dict(N=40, dZ=args.dz, cycles=60, eps_age=EPS_AGE, plasticity_rate=PLASTICITY_RATE,
                   n_substeps=N_SUBSTEPS, measure_every=20)
        plan = [(1, 0), (8, 0)]
        print("QUICK SMOKE TEST -- not real data.")
    else:
        cfg = dict(N=N, dZ=args.dz, cycles=CYCLES, eps_age=EPS_AGE, plasticity_rate=PLASTICITY_RATE,
                   n_substeps=N_SUBSTEPS, measure_every=MEASURE_EVERY)
        plan = [(nt, rep) for nt in TARGET_COUNTS for rep in range(args.n_realizations)]

    jobs = [(nt, rep, args.dz, cfg) for nt, rep in plan]
    n_workers = args.n_workers or os.cpu_count()
    print(f"{len(jobs)} realizations queued across {n_workers} workers "
          f"(dZ={args.dz}, N={cfg['N']}, cycles={cfg['cycles']}, "
          f"{args.n_realizations if not args.quick else 1} per target count)")

    t0 = time.time()
    done = 0
    with mp.Pool(n_workers) as pool, open(OUT_PATH, "a") as f:
        for result in pool.imap_unordered(run_one, jobs):
            f.write(json.dumps(result) + "\n")
            f.flush()
            done += 1
            elapsed = time.time() - t0
            eta = (len(jobs) - done) * elapsed / done / 60
            print(f"[{elapsed/60:6.1f} min] {done}/{len(jobs)} done "
                  f"(n_targets={result['n_targets']}, divergent={result['divergent']}) "
                  f"-- ETA {eta:.0f} min", flush=True)

    print(f"\nAll done in {(time.time()-t0)/60:.1f} min. Results in {OUT_PATH} ({done} realizations).")


if __name__ == "__main__":
    main()
