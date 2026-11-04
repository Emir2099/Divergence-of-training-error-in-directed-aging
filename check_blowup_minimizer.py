"""
check_blowup_minimizer.py

Re-runs ONE specific realization -- seed 9080010, the one that blew up to
error=16.49 by cycle 800 in calibration_results.jsonl (production-pair
setting: N=100, dZ=0.5, n_targets=8, eps_age=0.04, plasticity_rate=0.012) --
with logging added to every call training.py makes to
scipy.optimize.minimize, so we can see whether the relaxation step
(L-BFGS-B) ever failed to converge (res.success == False) around the point
the blow-up starts (cycle ~520 onward).

This does NOT change training.py on disk. It monkeypatches the
`scipy_minimize` name inside the already-imported `training` module at
runtime, so `training.minimize()` runs through exactly the same code path
it always does -- the only difference is each call's result is also
appended to MINIMIZE_LOG before being returned unchanged. The physics and
the resulting trajectory should come out bit-for-bit identical to what
calibration_results.jsonl already has for this seed; if they don't, that
itself would be a red flag worth knowing about.

This is ONE realization, not a sweep -- no multiprocessing, nothing else
to configure. Run in the same folder as spring_network.py and training.py:

    python check_blowup_minimizer.py

Prints a per-cycle summary of minimizer failures and writes the full
trajectory + log to blowup_diagnostic_9080010.json.
"""
import json
import time

import numpy as np

import training
from training import train_one_cycle, measure_error_and_order_parameter, lowest_modes_with_overlap
from spring_network import SpringNetwork

# -- exact config that produced seed 9080010 in calibration_results.jsonl --
N = 100
DZ = 0.5
N_TARGETS = 8
SEED = 9080010
EPS_AGE = 0.04
PLASTICITY_RATE = 0.012
N_SUBSTEPS = 8
MEASURE_EVERY = 20
CYCLES = 800

OUT_PATH = f"blowup_diagnostic_{SEED}.json"

# -- monkeypatch: wrap the scipy_minimize name training.py already imported,
# so training.minimize() is untouched but every call is logged. --
from scipy.optimize import minimize as _real_scipy_minimize

MINIMIZE_LOG = []
_current_cycle = {"cycle": -1, "phase": "setup"}  # updated by the loop below


def _logging_minimize(fun, x0, jac=None, method=None, options=None):
    t0 = time.time()
    res = _real_scipy_minimize(fun=fun, x0=x0, jac=jac, method=method, options=options)
    MINIMIZE_LOG.append({
        "cycle": _current_cycle["cycle"],
        "phase": _current_cycle["phase"],
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "nit": int(res.nit),
        "fun": float(res.fun),
        "dt": time.time() - t0,
    })
    return res


training.scipy_minimize = _logging_minimize


def main():
    rng = np.random.default_rng(SEED)
    net = SpringNetwork.random_diluted_delaunay(N=N, target_Z=4.0 + DZ, rng=rng)
    nodes = rng.permutation(N)
    source = tuple(nodes[:2])
    targets = [tuple(nodes[2 + 2 * t: 4 + 2 * t]) for t in range(N_TARGETS)]
    phases = rng.choice([-1.0, 1.0], size=N_TARGETS)

    traj = []
    t0 = time.time()
    for cycle in range(CYCLES + 1):
        if cycle % MEASURE_EVERY == 0:
            _current_cycle["cycle"], _current_cycle["phase"] = cycle, "measure"
            err, conv = measure_error_and_order_parameter(net, source, targets, phases)
            modes = lowest_modes_with_overlap(net, source, n_modes=6)
            traj.append({
                "cycle": cycle, "error": err, "converged": conv,
                "omega_trained": modes["omega_trained"], "omega_next": modes["omega_next"],
                "overlap": modes["overlap_trained"], "rank": modes["rank_of_trained"],
            })
            print(f"[{(time.time()-t0)/60:5.1f} min] cycle {cycle:4d}  error={err:8.4f}  "
                  f"failures_so_far={sum(1 for m in MINIMIZE_LOG if not m['success'])}", flush=True)
        if cycle < CYCLES:
            _current_cycle["cycle"], _current_cycle["phase"] = cycle, "train"
            train_one_cycle(net, source, targets, phases, eps_age=EPS_AGE,
                             plasticity_rate=PLASTICITY_RATE, n_substeps=N_SUBSTEPS)

    n_fail = sum(1 for m in MINIMIZE_LOG if not m["success"])
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. "
          f"{len(MINIMIZE_LOG)} total minimize() calls, {n_fail} failed to converge.")

    if n_fail:
        fail_cycles = sorted(set(m["cycle"] for m in MINIMIZE_LOG if not m["success"]))
        print(f"Failures occurred in cycles: {fail_cycles[:20]}"
              f"{' ...' if len(fail_cycles) > 20 else ''}")
        first_fail_cycle = fail_cycles[0]
        print(f"First failure at cycle {first_fail_cycle} "
              f"(error trajectory starts climbing sharply around cycle ~520-540, "
              f"per calibration_results.jsonl -- compare this number to that).")
    else:
        print("No minimizer failures at all -- every relaxation step reported "
              "res.success == True. The blow-up is NOT explained by a silently "
              "failed minimization; it's the actual plastic-update dynamics "
              "doing this, which points to a real (if rare) physical/numerical "
              "instability in the training rule itself, not a minimizer bug.")

    final_error = traj[-1]["error"]
    print(f"\nFinal error: {final_error:.4f} "
          f"(expect ~16.49 to match calibration_results.jsonl exactly -- if this "
          f"doesn't match, the environment/library versions differ enough to matter).")

    with open(OUT_PATH, "w") as f:
        json.dump({"seed": SEED, "n_targets": N_TARGETS, "trajectory": traj,
                   "minimize_log": MINIMIZE_LOG}, f, indent=2)
    print(f"Full trajectory + minimizer log written to {OUT_PATH}")


if __name__ == "__main__":
    main()