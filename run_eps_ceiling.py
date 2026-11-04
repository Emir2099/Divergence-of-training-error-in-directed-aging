"""
run_eps_ceiling.py -- do the 15 networks that did NOT diverge at eps=0.06 diverge at larger drive?

Follow-up showed: at eps=0.06, gamma=0.024 (dose ~4x seed 1000005's critical dose) only 1 of 16
networks diverged (seed 1000009). Two readings:
  (i)  every network has its own critical drive eps_c, and most are > 0.06 -> an eps_c
       distribution (a real threshold family) exists; look for a predictor of eps_c next.
  (ii) divergence is a property of rare networks -> no general threshold, no dynamics paper.
This pushes the other 15 to eps = 0.10 and 0.14 (same gamma=0.024, 400 cycles, measured every
10 cycles). 1000005 at eps=0.10 and 1000009 at eps=0.10 are positive controls (should spike).

Usage (same folder as run_smooth_ghost_rerun.py):
    python run_eps_ceiling.py --n-workers 16       # 33 runs, ~1 h on 16 cores
    python run_eps_ceiling.py --quick              # smoke test
Appends to eps_ceiling.jsonl.
"""
import argparse, json, multiprocessing as mp, os, time
from run_smooth_ghost_rerun import run_one, N, DZ, N_SUBSTEPS

OUT = "eps_ceiling.jsonl"
OTHER = [(2, 200002), (2, 200003), (2, 200004), (6, 600000), (6, 600009), (6, 600010), (7, 700001), (7, 700003), (7, 700009), (7, 700010), (8, 800004), (8, 800008), (9, 900007), (10, 1000002), (10, 1000009), (12, 1200004)]   # (n_targets, seed)
CONTROLS = [(10, 1000005), (10, 1000009)]


def mk(eps, g=0.024, cycles=400, every=10):
    return dict(N=N, dZ=DZ, cycles=cycles, eps_age=eps, plasticity_rate=g,
                n_substeps=N_SUBSTEPS, measure_every=every)


def job_fn(job):
    tag, nt, seed, cfg = job
    res = run_one((nt, seed, True, cfg))
    res.update(tag=tag, eps_age=cfg["eps_age"], gamma=cfg["plasticity_rate"], cycles=cfg["cycles"])
    return res


def build():
    jobs = [("ctrl", nt, s, mk(0.10)) for nt, s in CONTROLS]
    for nt, s in OTHER:
        if s == 1000009:
            continue
        for e in (0.10, 0.14):
            jobs.append(("scan", nt, s, mk(e)))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    jobs = build()
    if a.quick:
        jobs = jobs[:2]
        for j in jobs: j[3].update(N=40, cycles=40)
        print("QUICK SMOKE TEST -- not real data.")
    nw = a.n_workers or os.cpu_count()
    print(f"{len(jobs)} runs on {nw} workers", flush=True)
    t0 = time.time()
    with mp.Pool(nw) as pool, open(OUT, "a") as f:
        for i, r in enumerate(pool.imap_unordered(job_fn, jobs), 1):
            f.write(json.dumps(r) + "\n"); f.flush()
            print(f"[{(time.time()-t0)/60:6.1f} min] {i}/{len(jobs)} {r['tag']} seed={r['seed']} "
                  f"eps={r['eps_age']} peak/init={r['peak_over_init']:.2f}", flush=True)
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()