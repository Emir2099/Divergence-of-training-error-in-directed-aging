"""
run_eps_gamma_pilot.py -- pilot scan of the divergence over (eps_age, plasticity_rate)
for the two realizations whose divergence survives the smooth-ghost fix:
    seed 200005 (n_T=2) and seed 1000005 (n_T=10).
Same network / source / targets / phases for every grid point (same seed); only
eps_age and gamma change. Cycles scale as 9.6/gamma (constant aging "time", the
baseline gamma=0.012 -> 800 cycles) so slower aging is not cut short.

Question it answers: is there a boundary in (eps, gamma) separating spike from
no-spike, for BOTH seeds?  Contiguous region with an edge -> threshold exists.
Scattered / irregular -> no bifurcation result.

Usage (same folder as run_smooth_ghost_rerun.py):
    python run_eps_gamma_pilot.py               # 2 seeds x 5 eps x 5 gamma = 50 runs
    python run_eps_gamma_pilot.py --n-workers 16
    python run_eps_gamma_pilot.py --quick       # smoke test
Appends to eps_gamma_pilot.jsonl.
"""
import argparse, json, multiprocessing as mp, os, time
from run_smooth_ghost_rerun import run_one, N, DZ, N_SUBSTEPS

SEEDS = [(2, 200005, True), (10, 1000005, True)]
EPS = [0.02, 0.03, 0.04, 0.05, 0.06]
GAMMA = [0.006, 0.009, 0.012, 0.018, 0.024]
OUT = "eps_gamma_pilot.jsonl"


def job_fn(job):
    nt, seed, label, cfg = job
    res = run_one((nt, seed, label, cfg))
    res["eps_age"], res["gamma"], res["cycles"] = cfg["eps_age"], cfg["plasticity_rate"], cfg["cycles"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    jobs = []
    for nt, seed, lab in SEEDS:
        for e in EPS:
            for g in GAMMA:
                cfg = dict(N=N, dZ=DZ, cycles=int(round(9.6 / g)), eps_age=e, plasticity_rate=g,
                           n_substeps=N_SUBSTEPS, measure_every=20)
                jobs.append((nt, seed, lab, cfg))
    if a.quick:
        jobs = jobs[:2]
        for j in jobs: j[3].update(N=40, cycles=40)
        print("QUICK SMOKE TEST -- not real data.")
    # longest jobs first so the pool stays busy
    jobs.sort(key=lambda j: -j[3]["cycles"] * (1 + j[0]))
    nw = a.n_workers or os.cpu_count()
    print(f"{len(jobs)} runs on {nw} workers", flush=True)
    t0 = time.time()
    with mp.Pool(nw) as pool, open(OUT, "a") as f:
        for i, r in enumerate(pool.imap_unordered(job_fn, jobs), 1):
            f.write(json.dumps(r) + "\n"); f.flush()
            print(f"[{(time.time()-t0)/60:6.1f} min] {i}/{len(jobs)} seed={r['seed']} eps={r['eps_age']} "
                  f"gamma={r['gamma']} peak/init={r['peak_over_init']:.2f}", flush=True)
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
