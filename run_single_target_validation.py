"""
run_single_target_validation.py -- does this code train a SINGLE target well?

Why: across 101 realizations the median error reduction is only ~3% (final/init median 0.97);
n_T=1 gives final/init ~0.81. Raising the aging dose 4x-12x did not help. Before any claim about
trainability limits (n_T dependence, mode switching), the n_T=1 case must show strong training
(error falling toward ~0 for most networks), as in the published single-target result.

Runs the five n_T=1 realizations (same seeds/networks as Table 1) under three settings, each at
about 4x the baseline aging dose gamma*eps^2*cycles:
    S1: eps=0.04  gamma=0.012  3200 cycles   (4x longer)
    S2: eps=0.08  gamma=0.012   800 cycles   (4x via eps^2)
    S3: eps=0.04  gamma=0.048   800 cycles   (4x via gamma)
Measured every 100 cycles. Appends to single_target_validation.jsonl.

Usage (same folder as run_smooth_ghost_rerun.py):
    python run_single_target_validation.py --n-workers 15     # 15 runs, ~1.5-2 h
    python run_single_target_validation.py --quick            # smoke test
"""
import argparse, json, multiprocessing as mp, os, time
from run_smooth_ghost_rerun import run_one, N, DZ, N_SUBSTEPS
from run_solver_failure_check import REALIZATIONS

OUT = "single_target_validation.jsonl"
SETTINGS = {"S1": (0.04, 0.012, 3200), "S2": (0.08, 0.012, 800), "S3": (0.04, 0.048, 800)}


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
    for tag, (e, g, c) in SETTINGS.items():
        for nt, s, lab in seeds:
            cfg = dict(N=N, dZ=DZ, cycles=c, eps_age=e, plasticity_rate=g,
                       n_substeps=N_SUBSTEPS, measure_every=100)
            jobs.append((tag, nt, s, lab, cfg))
    if a.quick:
        jobs = jobs[:2]
        for j in jobs: j[4].update(N=40, cycles=40, measure_every=20)
        print("QUICK SMOKE TEST -- not real data.")
    jobs.sort(key=lambda j: -j[4]["cycles"])
    nw = a.n_workers or os.cpu_count()
    print(f"{len(jobs)} runs on {nw} workers", flush=True)
    t0 = time.time()
    with mp.Pool(nw) as pool, open(OUT, "a") as f:
        for i, r in enumerate(pool.imap_unordered(job_fn, jobs), 1):
            f.write(json.dumps(r) + "\n"); f.flush()
            best = min(p["error"] for p in r["trajectory"]) / r["init_error"]
            print(f"[{(time.time()-t0)/60:6.1f} min] {i}/{len(jobs)} {r['tag']} seed={r['seed']} "
                  f"final/init={r['final_error']/r['init_error']:.3f} best/init={best:.3f}", flush=True)
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()