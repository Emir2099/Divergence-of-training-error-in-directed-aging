"""
run_excluded_volume.py  --  COMPUTE ITEM 1: does the divergence survive when nodes cannot pass through each other?

Question.  The manuscript shows that three of four divergences are a collapse of a driven node pair (the normaliser of the
strain error goes to ~0) in a model whose nodes are points.  Here the SAME networks are aged with a short-range repulsion
between all node pairs (see cycle_tools.py: contact distance s_ij = min(sigma, 0.9 * initial distance), stiffness krep = 1)
and the same readouts are recorded.  sigma = 0 is the original model and must reproduce the old trajectories exactly.

Outcomes that this run can distinguish, for each network:
  (a) separations of the driven pairs stay finite and E no longer diverges   -> the divergence was a property of point nodes
  (b) separations shrink to the contact distance and E still grows            -> E needs a bounded normaliser even with excluded volume
  (c) separations stay finite but E / E0 still has an excursion               -> a genuine instability of the response

Nothing here changes the aging rule; sigma only enters the energy that the relaxation inside each cycle minimises.

Usage (PowerShell, from the folder that holds spring_network.py, training.py, cycle_tools.py, ...):
    python run_excluded_volume.py --quick                                   # 1-2 minute smoke test, N = 40
    python run_excluded_volume.py --nets 1000005,200005,700000 --sigmas 0,0.25,0.5,1.0 --n-workers 8
    python run_excluded_volume.py --nets 100001,100005 --sigmas 0.5,1.0 --n-workers 4       # optional, hours
Output: results_ev/*.jsonl (one file per job; a .done file marks a finished job; re-running the same command resumes).
Then:   python analyze_excluded_volume.py
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

NETS = {   # label: (seed, n_targets, eps, gamma, cycles, record every)
    "1000005": (1000005, 10, 0.04, 0.012, 800, 20),    # shortest target pair collapses near cycle 540 without excluded volume
    "200005":  (200005, 2, 0.04, 0.012, 400, 20),      # genuine transient near cycle 120
    "700000":  (700000, 7, 0.04, 0.012, 800, 20),      # control: no collapse in the original re-run
    "300003":  (300003, 3, 0.04, 0.012, 800, 20),      # control: transient of E at cycle 80, no collapse
    "1000009": (1000009, 10, 0.06, 0.024, 400, 20),    # second 10-target network that diverges (at eps = 0.06, gamma = 0.024)
    "100001":  (100001, 1, 0.03, 0.024, 4500, 50),     # single target, collapse of the target pair near cycle 4100
    "100005":  (100005, 1, 0.03, 0.024, 7600, 50),     # single target, collapse near cycle 7360
}


def run_job(job):
    import cycle_tools as ct
    label, sigma, krep, outdir, quick = job
    seed, nT, eps, gamma, cycles, every = NETS[label]
    N = 100
    if quick:
        N, cycles, every, nT = 40, 30, 10, min(nT, 3)
    tag = f"ev_{label}_s{sigma:g}" + ("_quick" if quick else "")
    t0 = time.time()
    ct.run_trajectory(tag, outdir, seed, nT, eps, gamma, cycles, every=every, sigma=sigma, krep=krep, N=N,
                      log=lambda m: print(m, flush=True), extra=dict(label=label))
    return tag, (time.time() - t0) / 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", default="1000005,200005,700000")
    ap.add_argument("--sigmas", default="0,0.25,0.5,1.0")
    ap.add_argument("--krep", type=float, default=1.0)
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--outdir", default="results_ev_2") # this was earlier used for normal dir which is correct too
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    nets = [n.strip() for n in a.nets.split(",") if n.strip()]
    sigmas = [float(x) for x in a.sigmas.split(",")]
    if a.quick:
        nets, sigmas, a.outdir = ["1000005"], [0.0, 0.5], "results_ev_quick"
        print("QUICK SMOKE TEST (N=40, 30 cycles) -- not real data.")
    for n in nets:
        if n not in NETS:
            sys.exit(f"unknown net {n}; choose from {list(NETS)}")
    nw = max(1, min(a.n_workers, 8))                      # more than 8 workers triggered the Windows 'paging file' error before
    jobs = [(n, s, a.krep, a.outdir, a.quick) for n in sorted(nets, key=lambda n: -NETS[n][4]) for s in sigmas]
    print(f"{len(jobs)} jobs on {nw} workers -> {a.outdir}/", flush=True)
    t0 = time.time()
    with mp.Pool(nw) as pool:
        for k, (tag, mins) in enumerate(pool.imap_unordered(run_job, jobs), 1):
            print(f"== finished {tag} in {mins:.1f} min   ({k}/{len(jobs)}, total {(time.time() - t0) / 60:.1f} min)", flush=True)
    print("All done. Next: python analyze_excluded_volume.py" + (" --dir results_ev_quick" if a.quick else ""))


if __name__ == "__main__":
    main()
