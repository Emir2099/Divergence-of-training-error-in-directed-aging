"""
run_production_sweep.py

Run this on your own machine, not in a hosted sandbox: it's built to run
for hours, using every CPU core you have, with no per-command time limit.

Strategy, based directly on the pilot result: n_targets 6-10 is where
omega/error decoupling showed up, so it gets more realizations than the
easier range, which mostly exists to confirm the mechanism holds there
the way it looked like it did in the pilot.

Usage:
    python run_production_sweep.py
    python run_production_sweep.py --n-workers 8      # override core count
    python run_production_sweep.py --quick             # small smoke test first

Results append to production_results.jsonl (one JSON line per realization)
as they complete, so you can inspect partial progress (or hand it to me)
at any time without stopping the run, and nothing is lost if it's killed.
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
EPS_AGE = 0.04
PLASTICITY_RATE = 0.012
N_SUBSTEPS = 8
MEASURE_EVERY = 20
CYCLES = 800          # generous budget; no 300s wall here

# (n_targets, n_realizations) -- extra depth in 6-10 where the pilot found
# the interesting behavior; enough elsewhere to confirm the easy-case story.
PLAN = [
    (1, 6), (2, 6), (3, 6), (4, 6), (5, 6),
    (6, 12), (7, 12), (8, 12), (9, 12), (10, 12),
    (12, 6), (15, 6),
]

OUT_PATH = "production_results.jsonl"


def run_one(job):
    n_targets, rep, cfg = job
    N, DZ, CYCLES = cfg["N"], cfg["dZ"], cfg["cycles"]
    EPS_AGE, PLASTICITY_RATE = cfg["eps_age"], cfg["plasticity_rate"]
    N_SUBSTEPS, MEASURE_EVERY = cfg["n_substeps"], cfg["measure_every"]

    seed = 100_000 * n_targets + rep
    rng = np.random.default_rng(seed)
    net = SpringNetwork.random_diluted_delaunay(N=N, target_Z=4.0 + DZ, rng=rng)
    nodes = rng.permutation(N)
    source = tuple(nodes[:2])
    targets = [tuple(nodes[2 + 2 * t: 4 + 2 * t]) for t in range(n_targets)]
    phases = rng.choice([-1.0, 1.0], size=n_targets)

    traj = []
    t0 = time.time()
    for cycle in range(CYCLES + 1):
        if cycle % MEASURE_EVERY == 0:
            err, conv = measure_error_and_order_parameter(net, source, targets, phases)
            modes = lowest_modes_with_overlap(net, source, n_modes=6)
            traj.append({
                "cycle": cycle, "error": err, "converged": conv,
                "omega_trained": modes["omega_trained"],
                "omega_next": modes["omega_next"],
                "overlap": modes["overlap_trained"],
                "rank": modes["rank_of_trained"],
            })
        if cycle < CYCLES:
            train_one_cycle(net, source, targets, phases, eps_age=EPS_AGE,
                             plasticity_rate=PLASTICITY_RATE, n_substeps=N_SUBSTEPS)

    return {
        "N": N, "dZ": DZ, "n_targets": n_targets, "delta": n_targets / N,
        "realization_seed": seed, "wall_time_s": time.time() - t0,
        "trajectory": traj,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=None,
                         help="default: all logical cores")
    parser.add_argument("--quick", action="store_true",
                         help="tiny smoke test (small N/cycles) to confirm the "
                              "environment works before committing hours to the real run")
    args = parser.parse_args()

    if args.quick:
        cfg = dict(N=40, dZ=DZ, cycles=60, eps_age=EPS_AGE,
                   plasticity_rate=PLASTICITY_RATE, n_substeps=N_SUBSTEPS,
                   measure_every=20)
        plan = [(1, 1), (8, 1)]
        print("QUICK SMOKE TEST MODE -- not real data, just checking it runs.")
    else:
        cfg = dict(N=N, dZ=DZ, cycles=CYCLES, eps_age=EPS_AGE,
                   plasticity_rate=PLASTICITY_RATE, n_substeps=N_SUBSTEPS,
                   measure_every=MEASURE_EVERY)
        plan = PLAN

    # cfg is now baked into every job tuple and pickled to each worker
    # explicitly -- this is what actually reaches spawned processes on
    # Windows/macOS, unlike a module-global overwritten in this (parent)
    # process only, which a fresh re-imported worker never sees.
    n_workers = args.n_workers or os.cpu_count()
    jobs = [(nt, rep, cfg) for nt, n_reps in plan for rep in range(n_reps)]
    print(f"{len(jobs)} realizations queued across {n_workers} workers "
          f"(N={cfg['N']}, dZ={cfg['dZ']}, cycles={cfg['cycles']})")

    t0 = time.time()
    done = 0
    # 'if __name__ == "__main__"' guard below is required on Windows for
    # multiprocessing to work at all -- don't move this call above it.
    with mp.Pool(n_workers) as pool, open(OUT_PATH, "a") as f:
        for result in pool.imap_unordered(run_one, jobs):
            f.write(json.dumps(result) + "\n")
            f.flush()
            done += 1
            elapsed = time.time() - t0
            rate = done / elapsed
            eta_min = (len(jobs) - done) / rate / 60 if rate > 0 else float("nan")
            print(f"[{elapsed/60:6.1f} min] {done}/{len(jobs)} done "
                  f"(n_targets={result['n_targets']}, final error="
                  f"{result['trajectory'][-1]['error']:.3f}) "
                  f"-- ETA {eta_min:.0f} min", flush=True)

    print(f"\nAll done in {(time.time()-t0)/60:.1f} minutes. "
          f"Results in {OUT_PATH} ({done} realizations).")


if __name__ == "__main__":
    main()
