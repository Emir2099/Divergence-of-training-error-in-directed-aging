"""
run_gentle_diagnostic.py -- is the weak / non-monotonic n_T=1 training a "too aggressive settings"
problem or a "training does not work in this code" problem?

Validation showed that at eps=0.04, gamma=0.012 (and 4x-dose variants) single-target error does
not fall monotonically: e.g. seed 100001 reached 0.33 at cycle 200 and was back to 0.94 at cycle
3200. If gentler aging (smaller strain amplitude eps and/or smaller rate gamma) gives a clean,
monotone decay of the error, the instability is a property of the aggressive regime and the code
can train. If error is still non-monotone / stalls at gentle settings, the training dynamics
themselves need to be debugged before any trainability claim.

Settings (n_T=1, five networks each, 1600 cycles, measured every 20):
    G1: eps=0.04 gamma=0.003   (same strain, 4x smaller rate)
    G2: eps=0.01 gamma=0.012   (4x smaller strain, same rate)
    G3: eps=0.01 gamma=0.003   (both)
Usage (same folder as run_smooth_ghost_rerun.py):
    python run_gentle_diagnostic.py --n-workers 15      # 15 runs, ~45-60 min
    python run_gentle_diagnostic.py --quick
Appends to gentle_diagnostic.jsonl.
"""
import argparse, json, multiprocessing as mp, os, time
from run_smooth_ghost_rerun import run_one, N, DZ, N_SUBSTEPS
from run_solver_failure_check import REALIZATIONS

OUT = "gentle_diagnostic.jsonl"
SETTINGS = {"G1": (0.04, 0.003), "G2": (0.01, 0.012), "G3": (0.01, 0.003)}
CYCLES = 1600


def job_fn(job):
    tag, nt, seed, label, cfg = job
    res = run_one((nt, seed, label, cfg))
    res.update(tag=tag, eps_age=cfg["eps_age"], gamma=cfg["plasticity_rate"], cycles=cfg["cycles"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    seeds = [(nt, s, lab) for nt, s, lab in REALIZATIONS if nt == 1]
    jobs = []
    for tag, (e, g) in SETTINGS.items():
        for nt, s, lab in seeds:
            cfg = dict(N=N, dZ=DZ, cycles=CYCLES, eps_age=e, plasticity_rate=g,
                       n_substeps=N_SUBSTEPS, measure_every=20)
            jobs.append((tag, nt, s, lab, cfg))
    if a.quick:
        jobs = jobs[:2]
        for j in jobs: j[4].update(N=40, cycles=40)
        print("QUICK SMOKE TEST -- not real data.")
    nw = a.n_workers or os.cpu_count()
    print(f"{len(jobs)} runs on {nw} workers", flush=True)
    t0 = time.time()
    with mp.Pool(nw) as pool, open(OUT, "a") as f:
        for i, r in enumerate(pool.imap_unordered(job_fn, jobs), 1):
            f.write(json.dumps(r) + "\n"); f.flush()
            errs = [p["error"] for p in r["trajectory"]]
            mono = sum(1 for x, y in zip(errs, errs[1:]) if y > x + 0.02)
            print(f"[{(time.time()-t0)/60:6.1f} min] {i}/{len(jobs)} {r['tag']} seed={r['seed']} "
                  f"final/init={r['final_error']/r['init_error']:.3f} best/init={min(errs)/r['init_error']:.3f} "
                  f"upticks>0.02={mono}", flush=True)
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()