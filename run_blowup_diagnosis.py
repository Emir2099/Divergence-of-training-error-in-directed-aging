"""
run_blowup_diagnosis.py -- what is the huge error (1e3-1e4) in the single-target blow-ups?

gentle_long showed n_T=1 blow-ups at eps=0.03: seed 100001 (error ~100 from cycle ~4200, up to
~1500) and seed 100005 (error 1e4 near cycle 7200-7500).  Candidate explanations:
  A  measurement artifact: training.measure_error_and_order_parameter / _driven_length still use a
     minimum-image length re-evaluated per call (my smooth-ghost fix only changed the DRIVING
     springs).  A length jump of O(1) at the cut locus gives error ~ 1/eps_measure = 1e3.
  B  a genuinely soft mode: aging produced a (near-)zero Hessian eigenvalue.  NOTE the earlier
     omega2_min column cannot show this: eigenvalues below 1e-8 are discarded as 'zero modes'.
  C  a snap-through / branch change of the relaxed state.

At every measurement in a window around the blow-up it records:
  err_orig   : the code's own measured error
  err_fixed  : same measurement but with the image of each target pair fixed at the baseline state
  cut_dist   : smallest distance of any separation component (source, targets) from L/2
  eig        : the 8 smallest Hessian eigenvalues at the undriven equilibrium (no 1e-8 cut)
  n_below_1e8 / 1e6 / 1e4 : number of eigenvalues below those values
Reading the output:
  err_orig huge but err_fixed ~ O(1)           -> A (artifact)
  err_orig ~ err_fixed huge, eig[2..] ~ 0       -> B (real soft mode)
  err_orig ~ err_fixed huge, eig fine           -> C (state change)

Usage (same folder as run_smooth_ghost_rerun.py):
    python run_blowup_diagnosis.py            # 2 processes, ~3-4 h
    python run_blowup_diagnosis.py --quick    # smoke test
Appends to blowup_diagnosis.jsonl.
"""
import argparse, json, multiprocessing as mp, time
import numpy as np

EPS, GAMMA, N, DZ, NSUB = 0.03, 0.024, 100, 0.5, 8
# (seed, total cycles, window start, window step)
JOBS = [(100001, 4400, 3300, 25), (100005, 7800, 6600, 25)]
OUT = "blowup_diagnosis.jsonl"


def diagnose(net, source, targets, phases, L, training, minimize):
    net.clear_ghosts()
    minimize(net)
    x_ref = net.pos.copy()
    # original measurement
    err_orig, _ = training.measure_error_and_order_parameter(net, source, targets, phases)
    net.pos = x_ref.copy()
    # fixed-image measurement
    net.clear_ghosts(); minimize(net); x_ref = net.pos.copy()
    shifts = [-np.round((net.pos[b] - net.pos[a]) / L) for a, b in targets]
    base = np.array([np.linalg.norm(net.pos[b] - net.pos[a] + s * L) for (a, b), s in zip(targets, shifts)])
    l0 = training._driven_length(net, source)
    net.set_ghost("source", source[0], source[1], 50.0, rest_length=l0 * (1 + 1e-3))
    minimize(net)
    drv = np.array([np.linalg.norm(net.pos[b] - net.pos[a] + s * L) for (a, b), s in zip(targets, shifts)])
    net.clear_ghosts(); net.pos = x_ref.copy()
    err_fixed = float(np.mean(np.abs((drv - base) / (base * 1e-3) - np.asarray(phases))))
    cut = min(abs(abs(c) - L / 2) for pair in [source] + list(targets)
              for c in (net.pos[pair[1]] - net.pos[pair[0]]))
    w = np.linalg.eigvalsh(net.hessian(net.pos, include_ghosts=False))
    return dict(err_orig=float(err_orig), err_fixed=err_fixed, cut_dist=float(cut),
                eig=[float(x) for x in w[:8]], n_below_1e8=int((w < 1e-8).sum()),
                n_below_1e6=int((w < 1e-6).sum()), n_below_1e4=int((w < 1e-4).sum()))


def run(job):
    import training
    from training import train_one_cycle, minimize
    from spring_network import SpringNetwork
    import run_smooth_ghost_rerun as rs
    rs.install_smooth_ghosts()
    seed, cycles, w0, step, quick = job
    rng = np.random.default_rng(seed)
    net = SpringNetwork.random_diluted_delaunay(N=(40 if quick else N), target_Z=4.0 + DZ, rng=rng)
    nn = 40 if quick else N
    nodes = rng.permutation(nn)
    source = tuple(nodes[:2]); targets = [tuple(nodes[2:4])]
    phases = rng.choice([-1.0, 1.0], size=1)
    rows, t0 = [], time.time()
    for c in range(cycles + 1):
        if (c >= w0 and (c - w0) % step == 0) or c % 500 == 0:
            d = diagnose(net, source, targets, phases, net.L, training, minimize)
            d.update(seed=seed, cycle=c, eps=EPS, gamma=GAMMA)
            with open(OUT, "a") as f:
                f.write(json.dumps(d) + "\n")
            print(f"[{(time.time()-t0)/60:6.1f} min] seed={seed} cycle={c} err_orig={d['err_orig']:.3g} "
                  f"err_fixed={d['err_fixed']:.3g} cut={d['cut_dist']:.3g} eig2={d['eig'][2]:.2e} "
                  f"n<1e-8:{d['n_below_1e8']}", flush=True)
        if c < cycles:
            train_one_cycle(net, source, targets, phases, eps_age=EPS, plasticity_rate=GAMMA, n_substeps=NSUB)
    return seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    jobs = [(s, c, w, st, a.quick) for s, c, w, st in JOBS]
    if a.quick:
        jobs = [(s, 30, 10, 10, True) for s, *_ in JOBS]
        print("QUICK SMOKE TEST -- not real data.")
    with mp.Pool(len(jobs)) as pool:
        pool.map(run, jobs)
    print("done ->", OUT)


if __name__ == "__main__":
    main()