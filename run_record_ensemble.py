"""
run_record_ensemble.py  --  COMPUTE ITEM 2: record the target separations, E0 and the gains for the whole ensemble.

IMPORTANT: this is a READOUT-ONLY re-recording.  E and E0 are only measurements; the aging rule never uses either of them, so
"training with E0" is not a different experiment.  What we can do is re-run the same trajectories and record, next to E, the
quantities needed to say what each excursion of E is: the separation of every target pair, its absolute response to the source
drive, and E0 (normalised by the initial separation).  Until now this was done for seven networks; this script does it for all
101 networks of the ensemble (eps = 0.04, gamma = 0.012, 800 cycles, every 20), with the same corrected ghost springs as the
paper.  The trajectories must reproduce the stored errors in data/s101.jsonl; analyze_ensemble_record.py checks this.

Optional --long: the 13 single-target runs of 8000 cycles (gamma = 0.024, eps = 0.01, 0.02, 0.03) that did NOT diverge,
re-run with the readouts recorded every 100 cycles.  This is expensive (about 90 minutes per run on one core of our container).

Usage (PowerShell, in the code folder):
    python run_record_ensemble.py --quick                       # 1-minute smoke test (N = 40)
    python run_record_ensemble.py --n-workers 8                 # all 101 networks; resumable (just run it again)
    python run_record_ensemble.py --limit 30 --n-workers 8      # fixed random subset of 30 (seeded), cheaper
    python run_record_ensemble.py --long --n-workers 8          # optional: 13 long single-target runs
Output: results_ens/*.jsonl, one file per network, *.done marks a finished one.
Then:   python analyze_ensemble_record.py --s101 ..\\data\\s101.jsonl     (the --s101 path is optional)
"""
import argparse
import multiprocessing as mp
import os
import random
import time


def run_job(job):
    import cycle_tools as ct
    tag, seed, nT, eps, gamma, cycles, every, N, outdir = job
    t0 = time.time()
    ct.run_trajectory(tag, outdir, seed, nT, eps, gamma, cycles, every=every, sigma=0.0, N=N,
                      log=lambda m: print(m, flush=True))
    return tag, (time.time() - t0) / 60


LONG_RUNS = [(100000, 0.01), (100000, 0.02), (100000, 0.03), (100001, 0.01), (100001, 0.02), (100003, 0.01), (100003, 0.02),
             (100003, 0.03), (100004, 0.01), (100004, 0.02), (100004, 0.03), (100005, 0.01), (100005, 0.02)]   # the 13 that did not diverge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--eps", type=float, default=0.04)
    ap.add_argument("--gamma", type=float, default=0.012)
    ap.add_argument("--cycles", type=int, default=800)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--long", action="store_true")
    ap.add_argument("--outdir", default="results_ens")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    from run_solver_failure_check import REALIZATIONS        # (n_targets, seed, label) for the 101 networks
    nw = max(1, min(a.n_workers, 8))
    if a.long:
        jobs = [(f"long_{s}_{e:g}", s, 1, e, 0.024, 8000, 100, 100, a.outdir) for s, e in LONG_RUNS]
    else:
        reals = list(REALIZATIONS)
        if a.limit:
            reals = random.Random(0).sample(reals, min(a.limit, len(reals)))
        reals.sort(key=lambda r: -r[0])                       # most targets first: best load balance
        jobs = [(f"ens_{s}", s, nT, a.eps, a.gamma, a.cycles, a.every, 100, a.outdir) for nT, s, _ in reals]
    if a.quick:
        jobs = [(t + "_quick", s, min(nT, 3), e, g, 30, 10, 40, "results_ens_quick") for t, s, nT, e, g, c, ev, N, o in jobs[:2]]
        nw = min(nw, 2)
        print("QUICK SMOKE TEST (N=40, 30 cycles, 2 networks) -- not real data.")
    print(f"{len(jobs)} jobs on {nw} workers -> {jobs[0][-1]}/", flush=True)
    t0 = time.time()
    with mp.Pool(nw) as pool:
        for k, (tag, mins) in enumerate(pool.imap_unordered(run_job, jobs), 1):
            print(f"== finished {tag} in {mins:.1f} min   ({k}/{len(jobs)}, total {(time.time() - t0) / 60:.1f} min)", flush=True)
    print("All done. Next: python analyze_ensemble_record.py" + (" --dir results_ens_quick" if a.quick else ""))


if __name__ == "__main__":
    main()
