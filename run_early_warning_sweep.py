"""
run_early_warning_sweep.py

Runs the first 100 cycles of every realization already classified as
blowup/stable from the production run, at fine-grained (every 10 cycle)
measurement, and tests whether early trajectory statistics (error-step
volatility, rank-switch count, max single-step omega jump) separate the two
groups. This does not depend on the day-0 static Hessian gap check
(check_initial_spectral_gap.py), which came back negative, it's a purely
dynamical, re-simulated check.

Run in the same folder as spring_network.py and training.py:
    python run_early_warning_sweep.py
    python run_early_warning_sweep.py --n-workers 8
    python run_early_warning_sweep.py --quick     (2-realization smoke test)

Appends to early_warning_results.jsonl as jobs complete.
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
N_CYCLES = 100
MEASURE_EVERY = 10

OUT_PATH = "early_warning_results.jsonl"

# (n_targets, seed, blowup) for all 101 realizations classified from the
# production run (blowup = error spiked well above baseline at some point
# during the full 800-cycle run this seed already completed).
REALIZATIONS = [
(1,100003,False),(1,100004,False),(1,100001,False),(1,100005,False),(1,100000,False),
(2,200002,False),(2,200004,False),(2,200001,False),(2,200000,False),(2,200005,True),
(3,300000,False),(3,300001,False),(3,300003,False),(2,200003,True),(3,300002,False),
(3,300004,False),(4,400000,False),(3,300005,False),(4,400003,False),(4,400002,False),
(5,500002,False),(5,500000,False),(4,400004,False),(5,500001,False),(5,500003,False),
(5,500005,True),(5,500004,True),(4,400001,False),(4,400005,False),(6,600000,True),
(6,600001,False),(6,600002,False),(6,600003,False),(6,600004,False),(6,600006,False),
(6,600005,True),(6,600007,True),(6,600008,False),(7,700000,False),(6,600009,True),
(6,600010,False),(6,600011,False),(7,700004,False),(7,700003,True),(7,700002,False),
(7,700006,True),(7,700010,False),(7,700005,True),(7,700001,True),(7,700009,True),
(7,700008,True),(8,800000,False),(7,700011,False),(7,700007,False),(8,800002,False),
(8,800006,False),(8,800007,False),(8,800003,False),(8,800001,False),(8,800005,True),
(8,800004,False),(8,800011,False),(8,800008,True),(9,900003,False),(9,900002,False),
(9,900000,False),(8,800009,False),(9,900001,False),(8,800010,False),(9,900005,False),
(9,900004,False),(9,900006,False),(9,900007,True),(9,900010,False),(9,900009,False),
(9,900008,False),(9,900011,False),(10,1000000,False),(10,1000002,False),(10,1000001,False),
(10,1000003,False),(10,1000005,True),(10,1000004,False),(10,1000007,True),(10,1000009,False),
(10,1000006,False),(10,1000008,False),(10,1000011,False),(10,1000010,False),(12,1200005,False),
(12,1200000,False),(12,1200003,False),(12,1200001,False),(15,1500000,False),(15,1500004,False),
(15,1500003,False),(15,1500002,False),(12,1200004,True),(15,1500005,False),(12,1200002,False),
(15,1500001,False),
]


def run_one(job):
    n_targets, seed, label, cfg = job
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
                "cycle": cycle, "error": err,
                "omega_trained": modes["omega_trained"],
                "omega_next": modes["omega_next"],
                "overlap": modes["overlap_trained"],
                "rank": modes["rank_of_trained"],
            })
        if cycle < cfg["cycles"]:
            train_one_cycle(net, source, targets, phases, eps_age=cfg["eps_age"],
                             plasticity_rate=cfg["plasticity_rate"], n_substeps=cfg["n_substeps"])

    ranks = [p["rank"] for p in traj]
    n_rank_switches = sum(1 for i in range(1, len(ranks)) if ranks[i] != ranks[i - 1])
    errs = np.array([p["error"] for p in traj])
    err_step_std = float(np.std(np.diff(errs)))
    omegas = np.array([p["omega_trained"] for p in traj])
    max_omega_jump = float(np.max(np.abs(np.diff(omegas))))

    return {
        "n_targets": n_targets, "seed": seed, "label": label,
        "wall_time_s": time.time() - t0,
        "n_rank_switches_first_100": n_rank_switches,
        "err_step_std_first_100": err_step_std,
        "max_omega_jump_first_100": max_omega_jump,
        "trajectory": traj,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=None,
                        help="worker processes (default: at most 4; lower this if memory is limited)")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.n_workers is not None and args.n_workers < 1:
        parser.error("--n-workers must be at least 1")

    cfg = dict(N=N, dZ=DZ, cycles=N_CYCLES, measure_every=MEASURE_EVERY,
               eps_age=EPS_AGE, plasticity_rate=PLASTICITY_RATE, n_substeps=N_SUBSTEPS)
    plan = REALIZATIONS[:2] if args.quick else REALIZATIONS
    if args.quick:
        cfg = dict(cfg); cfg["cycles"] = 20
        print("QUICK SMOKE TEST -- not real data.")

    # Each spawned worker imports NumPy/SciPy and their native DLLs. On
    # Windows, starting one worker per logical core can exhaust the paging
    # file before any jobs run, so keep the automatic pool deliberately small.
    n_workers = args.n_workers or min(os.cpu_count() or 1, 4)
    jobs = [(nt, seed, ("blowup" if b else "stable"), cfg) for nt, seed, b in plan]
    print(f"{len(jobs)} realizations queued across {n_workers} workers, {cfg['cycles']} cycles each")

    t0 = time.time()
    done = 0
    label_counts = {"blowup": [], "stable": []}
    with mp.Pool(n_workers) as pool, open(OUT_PATH, "a") as f:
        for result in pool.imap_unordered(run_one, jobs):
            f.write(json.dumps(result) + "\n")
            f.flush()
            done += 1
            label_counts[result["label"]].append(result["err_step_std_first_100"])
            print(f"[{(time.time()-t0)/60:5.1f} min] {done}/{len(jobs)}  "
                  f"{result['label']:>7} n_targets={result['n_targets']} "
                  f"err_step_std={result['err_step_std_first_100']:.4f}  "
                  f"rank_switches={result['n_rank_switches_first_100']}", flush=True)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min.")
    for label in ("stable", "blowup"):
        vals = np.array(label_counts[label])
        if len(vals):
            print(f"{label}: n={len(vals)}  err_step_std mean={vals.mean():.4f}  "
                  f"sem={vals.std(ddof=1)/np.sqrt(len(vals)) if len(vals)>1 else float('nan'):.4f}")


if __name__ == "__main__":
    main()