"""
run_dose_followup.py -- tests the "aging dose" rule found in the (eps, gamma) pilot.

Pilot finding (seed 1000005): onset cycle * gamma * eps^2 = 0.0090 +/- 5% over 13 spiking runs.
Hypothesis: divergence happens when accumulated aging dose D = gamma * eps^2 * cycles reaches a
network-specific critical value D_c (D_c ~ 0.009 for seed 1000005). The pilot's "no spike at
eps <= 0.03" is then just a too-short window (tau = gamma*cycles was capped at 9.6).

Three tests:
  A  long window, seed 1000005: eps=0.03 (tau up to 16; predicted onset tau ~ 10) and
     eps=0.02 (tau up to 32; predicted onset ~ 22). If they spike near the predicted tau -> rule
     holds and there is NO eps threshold, only a delay.
  B  gamma boundary at eps=0.04: gamma = 0.013..0.016 (pilot: spike at 0.012, none at 0.018).
  C  generality: 16 other networks (non-diverging at baseline) at eps=0.06, gamma=0.024, 400 cycles
     (dose 0.035, ~4x the seed-1000005 critical dose). If many diverge -> divergence is generic,
     with a distribution of D_c across networks.

Usage (same folder as run_smooth_ghost_rerun.py):
    python run_dose_followup.py --n-workers 16        # ~1.5 h on 16 cores
    python run_dose_followup.py --quick               # smoke test
Appends to dose_followup.jsonl.
"""
import argparse, json, multiprocessing as mp, os, time
from run_smooth_ghost_rerun import run_one, N, DZ, N_SUBSTEPS

OUT = "dose_followup.jsonl"
OTHER = [(2, 200002), (2, 200003), (2, 200004), (6, 600000), (6, 600009), (6, 600010), (7, 700001), (7, 700003), (7, 700009), (7, 700010), (8, 800004), (8, 800008), (9, 900007), (10, 1000002), (10, 1000009), (12, 1200004)]   # (n_targets, seed), non-diverging at baseline in the smooth rerun


def mk(eps, g, cycles, every=20):
    return dict(N=N, dZ=DZ, cycles=cycles, eps_age=eps, plasticity_rate=g,
                n_substeps=N_SUBSTEPS, measure_every=every)


def job_fn(job):
    tag, nt, seed, cfg = job
    res = run_one((nt, seed, True, cfg))
    res.update(tag=tag, eps_age=cfg["eps_age"], gamma=cfg["plasticity_rate"], cycles=cfg["cycles"])
    return res


def build():
    jobs = []
    jobs.append(("A", 10, 1000005, mk(0.03, 0.024, 667)))
    jobs.append(("A", 10, 1000005, mk(0.03, 0.012, 1333)))
    jobs.append(("A", 10, 1000005, mk(0.02, 0.024, 1333)))
    for g in (0.013, 0.014, 0.015, 0.016):
        jobs.append(("B", 10, 1000005, mk(0.04, g, int(round(9.6 / g)))))
    for nt, seed in OTHER:
        jobs.append(("C", nt, seed, mk(0.06, 0.024, 400)))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    jobs = build()
    if a.quick:
        jobs = [jobs[0], jobs[-1]]
        for j in jobs: j[3].update(N=40, cycles=40)
        print("QUICK SMOKE TEST -- not real data.")
    jobs.sort(key=lambda j: -j[3]["cycles"])
    nw = a.n_workers or os.cpu_count()
    print(f"{len(jobs)} runs on {nw} workers", flush=True)
    t0 = time.time()
    with mp.Pool(nw) as pool, open(OUT, "a") as f:
        for i, r in enumerate(pool.imap_unordered(job_fn, jobs), 1):
            f.write(json.dumps(r) + "\n"); f.flush()
            print(f"[{(time.time()-t0)/60:6.1f} min] {i}/{len(jobs)} {r['tag']} seed={r['seed']} "
                  f"eps={r['eps_age']} gamma={r['gamma']} peak/init={r['peak_over_init']:.2f}", flush=True)
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()