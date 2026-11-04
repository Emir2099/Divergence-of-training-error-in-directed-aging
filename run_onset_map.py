"""
run_onset_map.py  --  COMPUTE ITEM 4: does the onset regularity  n* ~ D / (gamma * eps^2)  hold in a SECOND network?

In seed 1000005 the first measured cycle n* at which E exceeds five times its initial value follows n* ~ D/(gamma eps^2), D = 0.0096.
Here another network is aged over a grid of (eps, gamma) and the same quantity is measured, together with the target separations,
E0 and the gains, so that every divergence found can be classified (collapse of a target pair, or not).

Design
  * grid: eps x gamma (default eps in {0.03, 0.04, 0.05, 0.06, 0.08}, gamma in {0.012, 0.018, 0.024});
  * budget: n_max = min(max_cycles, budget_ng / gamma) cycles, as in the original map (n * gamma ~ 9.6, default);
  * a run is stopped `--extra` cycles (default 100) after E first exceeds 5 x E(0) -- the divergence is already established and
    the target separations after it are recorded; runs that never diverge use the whole budget;
  * the readouts are recorded every `every` cycles (default 10, so that small n* are resolved to 10 cycles).

Suggested second networks (neither was part of the original map)
  seed 1000009, 10 targets  : diverged at eps = 0.06, gamma = 0.024 within 400 cycles (42x)       ->  --seed 1000009 --n-targets 10
  seed 100001, 1 target     : collapse at eps = 0.03, gamma = 0.024 after ~4100 cycles (D ~ 0.09)  ->  --seed 100001 --n-targets 1
                                --budget-ng 100 --max-cycles 4500 --eps-list 0.04,0.05,0.06 --gamma-list 0.012,0.024

Usage (PowerShell, in the code folder):
    python run_onset_map.py --quick                                              # 1-2 minute smoke test, N = 40
    python run_onset_map.py --seed 1000009 --n-targets 10 --n-workers 8         # ~2-3 CPU-hours, 15 runs; resumable
    python run_onset_map.py --seed 100001 --n-targets 1 --budget-ng 100 --max-cycles 4500 --eps-list 0.04,0.05,0.06 --gamma-list 0.012,0.024 --n-workers 6
Output: results_onset/onset_<seed>_e<eps>_g<gamma>.jsonl (+ .done).  Then:  python analyze_onset_map.py
"""
import argparse
import multiprocessing as mp
import os
import time


def make_stop(factor, extra):
    def stop(cycles, errors):
        first = next((c for c, e in zip(cycles, errors) if e > factor * errors[0]), None)
        return first is not None and cycles[-1] >= first + extra
    return stop


def run_job(job):
    import cycle_tools as ct
    seed, nT, eps, gamma, n_max, every, extra, N, outdir = job
    tag = f"onset_{seed}_e{eps:g}_g{gamma:g}"
    t0 = time.time()
    ct.run_trajectory(tag, outdir, seed, nT, eps, gamma, n_max, every=every, N=N, log=lambda m: print(m, flush=True),
                      stop=make_stop(5.0, extra), ckpt_every=100)
    return tag, (time.time() - t0) / 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1000009)
    ap.add_argument("--n-targets", type=int, default=10)
    ap.add_argument("--eps-list", default="0.03,0.04,0.05,0.06,0.08")
    ap.add_argument("--gamma-list", default="0.012,0.018,0.024")
    ap.add_argument("--budget-ng", type=float, default=9.6)
    ap.add_argument("--max-cycles", type=int, default=2000)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--extra", type=int, default=100)
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--outdir", default="results_onset")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    eps_l = [float(x) for x in a.eps_list.split(",")]
    gam_l = [float(x) for x in a.gamma_list.split(",")]
    N = 100
    if a.quick:
        eps_l, gam_l, N, a.n_targets, a.extra, a.outdir = [0.06, 0.1], [0.024], 40, 3, 20, "results_onset_quick"
        a.budget_ng, a.every = 0.6, 5          # 25 cycles
        print("QUICK SMOKE TEST (N=40, 25 cycles) -- not real data.")
    nw = max(1, min(a.n_workers, 8))
    jobs = []
    for e in eps_l:
        for g in gam_l:
            n_max = int(min(a.max_cycles, round(a.budget_ng / g)))
            jobs.append((a.seed, a.n_targets, e, g, n_max, a.every, a.extra, N, a.outdir))
    jobs.sort(key=lambda j: -j[4] * 1.0)                           # longest first
    print(f"{len(jobs)} runs on {nw} workers -> {a.outdir}/   (cycle budgets {sorted({j[4] for j in jobs})})", flush=True)
    t0 = time.time()
    with mp.Pool(nw) as pool:
        for k, (tag, mins) in enumerate(pool.imap_unordered(run_job, jobs), 1):
            print(f"== finished {tag} in {mins:.1f} min   ({k}/{len(jobs)}, total {(time.time() - t0) / 60:.1f} min)", flush=True)
    print("All done. Next: python analyze_onset_map.py" + (" --dir results_onset_quick" if a.quick else ""))


if __name__ == "__main__":
    main()
