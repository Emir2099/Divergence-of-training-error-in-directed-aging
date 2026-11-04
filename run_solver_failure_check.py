"""
run_solver_failure_check.py

The systematic version of the one-realization check in
check_blowup_minimizer.py. Tests whether relaxation-solver (L-BFGS-B)
non-convergence predicts which realizations diverge, using the SAME 101
realizations (same seeds, same network, same n_targets) that underlie
Table 1 of the manuscript -- so this is directly comparable to the
existing structural (day-0 spectrum) and dynamical (first-100-cycle
volatility) tests in Section III.E, not a new dataset.

This is the single most likely thing a referee asks for, given the
manuscript currently presents the solver-failure mechanism as evidence
from one realization. This script turns that into a real test.

COST: this re-simulates each realization at the FULL 800-cycle depth
(not just the first 100, since the one case already examined showed the
failure-rate ramp starting around cycle 300-500, well past the window
the existing early-volatility test covers). That makes it comparable in
cost to the original run_production_sweep.py. Two ways to control that:

  --n-stable N   use all 20 divergent realizations (fixed) plus a random
                 sample of N stable ones, instead of all 81. Default 20
                 (40 total jobs, roughly 40/102 of the original sweep's
                 cost). Pass a larger number, or --n-stable 81 for the
                 full matched set (101 total, same cost as the original
                 production sweep).

Usage (same folder as spring_network.py and training.py):
    python run_solver_failure_check.py
    python run_solver_failure_check.py --n-stable 81   # full set, Table-1-identical
    python run_solver_failure_check.py --n-workers 8
    python run_solver_failure_check.py --quick           # 2-realization smoke test

Appends to solver_failure_results.jsonl as jobs complete.
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import time

import numpy as np

N = 100
DZ = 0.5
EPS_AGE = 0.04
PLASTICITY_RATE = 0.012
N_SUBSTEPS = 8
MEASURE_EVERY = 20
CYCLES = 800

OUT_PATH = "solver_failure_results.jsonl"

# Identical to the REALIZATIONS list in run_early_warning_sweep.py -- same
# 101 (n_targets, seed, label) triples underlying Table 1. Kept here
# verbatim so this script is self-contained and provably matches.
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
    # Each worker process imports fresh and patches its OWN copy of the
    # training module -- this must happen inside the worker, not the
    # parent, for the patch to take effect after multiprocessing spawns.
    import training
    from training import train_one_cycle, measure_error_and_order_parameter, lowest_modes_with_overlap
    from spring_network import SpringNetwork
    from scipy.optimize import minimize as real_scipy_minimize

    n_targets, seed, label, cfg = job
    minimize_log = []
    current = {"cycle": -1}

    def logging_minimize(fun, x0, jac=None, method=None, options=None):
        res = real_scipy_minimize(fun=fun, x0=x0, jac=jac, method=method, options=options)
        minimize_log.append((current["cycle"], bool(res.success)))
        return res

    training.scipy_minimize = logging_minimize

    rng = np.random.default_rng(seed)
    net = SpringNetwork.random_diluted_delaunay(N=cfg["N"], target_Z=4.0 + cfg["dZ"], rng=rng)
    nodes = rng.permutation(cfg["N"])
    source = tuple(nodes[:2])
    targets = [tuple(nodes[2 + 2 * t: 4 + 2 * t]) for t in range(n_targets)]
    phases = rng.choice([-1.0, 1.0], size=n_targets)

    traj = []
    t0 = time.time()
    for cycle in range(cfg["cycles"] + 1):
        current["cycle"] = cycle
        if cycle % cfg["measure_every"] == 0:
            err, conv = measure_error_and_order_parameter(net, source, targets, phases)
            modes = lowest_modes_with_overlap(net, source, n_modes=6)
            traj.append({"cycle": cycle, "error": err, "converged": conv,
                         "omega_trained": modes["omega_trained"], "rank": modes["rank_of_trained"]})
        if cycle < cfg["cycles"]:
            train_one_cycle(net, source, targets, phases, eps_age=cfg["eps_age"],
                             plasticity_rate=cfg["plasticity_rate"], n_substeps=cfg["n_substeps"])

    n_calls = len(minimize_log)
    n_fail = sum(1 for _, ok in minimize_log if not ok)
    # failures binned into 100-cycle windows, so this is directly comparable
    # to the existing "first 100 cycles" early-volatility test
    fail_by_window = {}
    for cyc, ok in minimize_log:
        if not ok:
            w = (cyc // 100) * 100
            fail_by_window[w] = fail_by_window.get(w, 0) + 1

    return {
        "n_targets": n_targets, "seed": seed, "label": "blowup" if label else "stable",
        "wall_time_s": time.time() - t0,
        "n_minimize_calls": n_calls, "n_minimize_failures": n_fail,
        "failure_rate": n_fail / n_calls if n_calls else 0.0,
        "failures_by_100cycle_window": fail_by_window,
        "final_error": traj[-1]["error"],
        "peak_error": max(p["error"] for p in traj),
        "trajectory": traj,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--n-stable", type=int, default=20,
                         help="how many stable realizations to include alongside all 20 "
                              "divergent ones (default 20, i.e. 40 total jobs; pass 81 "
                              "for the full Table-1-identical set, 101 total jobs)")
    parser.add_argument("--select-seed", type=int, default=0,
                         help="seed for randomly choosing which stable realizations to include")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    divergent = [(nt, s, lab) for nt, s, lab in REALIZATIONS if lab]
    stable = [(nt, s, lab) for nt, s, lab in REALIZATIONS if not lab]
    assert len(divergent) == 20 and len(stable) == 81

    n_stable = min(args.n_stable, len(stable))
    rnd = random.Random(args.select_seed)
    chosen_stable = rnd.sample(stable, n_stable)
    selection = divergent + chosen_stable

    if args.quick:
        cfg = dict(N=40, dZ=DZ, cycles=60, eps_age=EPS_AGE, plasticity_rate=PLASTICITY_RATE,
                   n_substeps=N_SUBSTEPS, measure_every=20)
        selection = selection[:2]
        print("QUICK SMOKE TEST -- not real data.")
    else:
        cfg = dict(N=N, dZ=DZ, cycles=CYCLES, eps_age=EPS_AGE, plasticity_rate=PLASTICITY_RATE,
                   n_substeps=N_SUBSTEPS, measure_every=MEASURE_EVERY)

    jobs = [(nt, seed, label, cfg) for nt, seed, label in selection]
    n_workers = args.n_workers or os.cpu_count()
    print(f"{len(jobs)} realizations queued across {n_workers} workers "
          f"({sum(1 for j in jobs if j[2])} divergent + {sum(1 for j in jobs if not j[2])} stable), "
          f"N={cfg['N']}, cycles={cfg['cycles']}")

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
                  f"({result['label']}, n_targets={result['n_targets']}, "
                  f"failure_rate={result['failure_rate']:.3f}) -- ETA {eta:.0f} min", flush=True)

    print(f"\nAll done in {(time.time()-t0)/60:.1f} min. Results in {OUT_PATH} ({done} realizations).")


if __name__ == "__main__":
    main()
