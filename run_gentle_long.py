"""
run_gentle_long.py -- long single-target runs in the gentle regime.

Gentle diagnostic result: at eps=0.01 training is smooth and monotone (seed 100001 error fell
0.91 -> 0.32 over 1600 cycles, trained mode softened 0.087 -> 0.042, rank 0 throughout), but most
networks had not trained yet (too little total aging). At eps=0.04 error is non-monotone
(seed 100001 shows 14 upticks even at gamma=0.003). The published method runs 1e4-2e4 cycles.

This run asks, for the five n_T=1 networks:
  (a) does the error keep falling monotonically toward low values in the gentle regime?
  (b) at which strain amplitude eps does non-monotone behaviour start (per network)?
  (c) does the trained mode switch rank during clean, monotone training?

Settings: gamma=0.024, 8000 cycles, measured every 100, eps in {0.01, 0.02, 0.03}.
15 runs; ~3-4 h on 15 workers.  Appends to gentle_long.jsonl.
Usage (same folder as run_smooth_ghost_rerun.py):
    python run_gentle_long.py --n-workers 15
    python run_gentle_long.py --quick
"""
import argparse, json, multiprocessing as mp, os, time
from run_smooth_ghost_rerun import run_one, N, DZ, N_SUBSTEPS
from run_solver_failure_check import REALIZATIONS

OUT = "gentle_long.jsonl"
EPS = [0.01, 0.02, 0.03]
GAMMA = 0.024
CYCLES = 8000


def job_fn(job):
    nt, seed, label, cfg = job
    res = run_one((nt, seed, label, cfg))
    res.update(eps_age=cfg["eps_age"], gamma=cfg["plasticity_rate"], cycles=cfg["cycles"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    seeds = [(nt, s, lab) for nt, s, lab in REALIZATIONS if nt == 1]
    jobs = []
    for e in EPS:
        for nt, s, lab in seeds:
            cfg = dict(N=N, dZ=DZ, cycles=CYCLES, eps_age=e, plasticity_rate=GAMMA,
                       n_substeps=N_SUBSTEPS, measure_every=100)
            jobs.append((nt, s, lab, cfg))
    if a.quick:
        jobs = jobs[:2]
        for j in jobs: j[3].update(N=40, cycles=40, measure_every=20)
        print("QUICK SMOKE TEST -- not real data.")
    nw = a.n_workers or os.cpu_count()
    print(f"{len(jobs)} runs on {nw} workers", flush=True)
    t0 = time.time()
    with mp.Pool(nw) as pool, open(OUT, "a") as f:
        for i, r in enumerate(pool.imap_unordered(job_fn, jobs), 1):
            f.write(json.dumps(r) + "\n"); f.flush()
            errs = [p["error"] for p in r["trajectory"]]
            up = sum(1 for x, y in zip(errs, errs[1:]) if y > x + 0.02)
            print(f"[{(time.time()-t0)/60:6.1f} min] {i}/{len(jobs)} seed={r['seed']} eps={r['eps_age']} "
                  f"final/init={r['final_error']/r['init_error']:.3f} best/init={min(errs)/r['init_error']:.3f} "
                  f"upticks>0.02={up}", flush=True)
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()