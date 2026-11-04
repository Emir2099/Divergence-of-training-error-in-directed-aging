"""
run_sensitivity.py -- is the single-target blow-up a deterministic property of (network, eps, gamma),
or does it depend on tiny perturbations?

blowup_diagnosis showed (eps=0.03, gamma=0.024):
  seed 100001: blow-up reproduced (error ~105 from cycle ~4175, healthy Hessian spectrum)
  seed 100005: NOT reproduced (gentle_long: error 6.9 -> 11590 at cycle 7200-7500; diagnosis run: 2.46 at 7800)
The two runs differ only in how often the state was measured (each measurement re-minimizes, i.e.
nudges the state at solver-tolerance level) -> suggests sensitive dependence on tiny perturbations.

This run repeats each network with a relative random perturbation of the initial rest lengths of
size delta = 0 (control), 1e-12, 1e-10, 1e-8, 1e-6, 1e-4 (100005 also 1e-11, 1e-9), with the SAME
measurement schedule (error every 100 cycles) for every replica.
Questions: (1) does delta=0 reproduce itself? (2) how fast do replicas separate? (3) which
fraction blow up (error > 20x initial) and when?

Usage (same folder as run_smooth_ghost_rerun.py):
    python run_sensitivity.py --reduced --n-workers 8   # 8 runs, ~3 h  (recommended if memory is tight)
    python run_sensitivity.py --n-workers 14            # 14 runs, ~3 h, needs more free RAM / paging file
    python run_sensitivity.py --quick
Appends to sensitivity.jsonl.
"""
import os
# one math-library thread per worker: many workers x many BLAS threads exhausted the Windows
# paging file ("DLL load failed ... paging file is too small")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, json, multiprocessing as mp, time
import numpy as np

EPS, GAMMA, N, DZ, NSUB = 0.03, 0.024, 100, 0.5, 8
OUT = "sensitivity.jsonl"
JOBS = [(100001, 5000, d) for d in (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4)] + \
       [(100005, 8000, d) for d in (0.0, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-6, 1e-4)]


def run(job):
    import training
    from training import train_one_cycle, measure_error_and_order_parameter
    from spring_network import SpringNetwork
    import run_smooth_ghost_rerun as rs
    rs.install_smooth_ghosts()
    seed, cycles, delta, quick = job
    n = 40 if quick else N
    rng = np.random.default_rng(seed)
    net = SpringNetwork.random_diluted_delaunay(N=n, target_Z=4.0 + DZ, rng=rng)
    nodes = rng.permutation(n)
    source = tuple(nodes[:2]); targets = [tuple(nodes[2:4])]
    phases = rng.choice([-1.0, 1.0], size=1)
    if delta > 0:
        pr = np.random.default_rng(777)
        net.rest_length = net.rest_length * (1 + delta * pr.standard_normal(len(net.rest_length)))
    traj, t0 = [], time.time()
    for c in range(cycles + 1):
        if c % (10 if quick else 100) == 0:
            e, _ = measure_error_and_order_parameter(net, source, targets, phases)
            traj.append((c, float(e)))
        if c < cycles:
            train_one_cycle(net, source, targets, phases, eps_age=EPS, plasticity_rate=GAMMA, n_substeps=NSUB)
    init = traj[0][1]
    blow = next((c for c, e in traj if e > 20 * init), None)
    return dict(seed=seed, delta=delta, cycles=cycles, init_error=init, traj=traj,
                blowup_cycle=blow, peak_over_init=max(e for _, e in traj) / init,
                wall_s=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reduced", action="store_true",
                    help="8 runs (deltas 0, 1e-10, 1e-8, 1e-6 for both seeds); use with --n-workers 8")
    a = ap.parse_args()
    keep = (0.0, 1e-10, 1e-8, 1e-6)
    jobs = [(s, c, d, a.quick) for s, c, d in JOBS if (not a.reduced) or d in keep]
    if a.quick:
        jobs = [(100001, 30, 0.0, True), (100005, 30, 1e-8, True)]
        print("QUICK SMOKE TEST -- not real data.")
    jobs.sort(key=lambda j: -j[1])
    nw = a.n_workers or os.cpu_count()
    print(f"{len(jobs)} runs on {nw} workers", flush=True)
    t0 = time.time()
    with mp.Pool(nw) as pool, open(OUT, "a") as f:
        for i, r in enumerate(pool.imap_unordered(run, jobs), 1):
            f.write(json.dumps(r) + "\n"); f.flush()
            print(f"[{(time.time()-t0)/60:6.1f} min] {i}/{len(jobs)} seed={r['seed']} delta={r['delta']:g} "
                  f"peak/init={r['peak_over_init']:.2f} blowup_cycle={r['blowup_cycle']}", flush=True)
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()