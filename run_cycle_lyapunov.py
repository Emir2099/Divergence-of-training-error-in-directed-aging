"""
run_cycle_lyapunov.py  --  COMPUTE ITEM 3: finite-time growth rates of the full cycle map Phi on the rest-length vector.

What is computed.  Phi maps the rest lengths l0 (about 225 numbers) to the rest lengths after one complete strain cycle
(0 -> +eps -> 0 -> -eps -> 0, eight relax-and-update substeps, ghost springs as in the paper).  Along the aging trajectory
of one network we follow K nearby copies whose rest lengths differ by delta * v_k (v_k orthonormal) and re-orthonormalise
them after every cycle (Benettin / QR method).  The log of the diagonal of R is the growth of the k-th direction in that cycle;
its average over a window of cycles is the finite-time growth rate lambda_k (per cycle).  lambda_1 > 0 means that a
perturbation of the rest lengths grows under the cycle map over that window.  Every `every` cycles the usual readouts are
recorded too (target separations, E, E0, gains) together with lambda_max(S) of the undriven network, so that the growth rate of
the actual map can be set against the undriven linear criterion of the paper.

The perturbed copies are restarted from the unperturbed positions, so every copy follows the same branch of equilibria.
Forward differences with delta = 1e-4 (norm of the perturbation of the whole vector, i.e. ~7e-6 per bond); run --delta-check first
to see that the result does not depend on delta (the solver noise limits small delta, the nonlinearity limits large delta).

Cost: (K + 1) times one ordinary trajectory.  With K = 3 and our container: about 5 s per cycle for a 10-target network.

Usage (PowerShell, in the code folder):
    python run_cycle_lyapunov.py --quick                                   # 1-2 minute smoke test, N = 40
    python run_cycle_lyapunov.py --delta-check --nets 1000005              # 3-5 minutes: is delta = 1e-4 acceptable?
    python run_cycle_lyapunov.py --nets 1000005,200005,700000 --n-workers 3
    python run_cycle_lyapunov.py --nets 100001 --K 2                       # optional, single-target network, ~4500 cycles
Output: results_cyc/cyc_<net>.jsonl ("step" records: one per cycle; "meas" records: every `every` cycles).  Resumable.
Then:   python analyze_cycle_lyapunov.py
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np


def phi(net, source, targets, phases, eps, gamma, nsub, pos, l0):
    """One aging cycle from the state (pos, l0). Returns the new (pos, l0)."""
    from training import train_one_cycle
    net.pos = pos.copy()
    net.rest_length = l0.copy()
    net.clear_ghosts()
    train_one_cycle(net, source, targets, phases, eps_age=eps, plasticity_rate=gamma, n_substeps=nsub)
    return net.pos.copy(), net.rest_length.copy()


def setup(label, quick, sigma):
    import cycle_tools as ct
    from run_excluded_volume import NETS
    seed, nT, eps, gamma, cycles, every = NETS[label]
    N = 100
    if quick:
        N, nT, cycles, every = 40, min(nT, 3), 20, 10
    net, source, targets, phases = ct.build(seed, nT, N)
    ct.attach_excluded_volume(net, sigma)
    return ct, net, source, targets, phases, seed, nT, eps, gamma, cycles, every


def run_net(job):
    label, K, delta, sigma, outdir, quick, cycles_arg, every_arg = job
    ct, net, source, targets, phases, seed, nT, eps, gamma, cycles, every = setup(label, quick, sigma)
    cycles = cycles_arg or cycles
    every = every_arg or every
    nsub = ct.NSUB_DEFAULT
    nb = len(net.rest_length)
    tag = f"cyc_{label}" + (f"_s{sigma:g}" if sigma else "") + ("_quick" if quick else "")
    os.makedirs(outdir, exist_ok=True)
    path, done, ck = [os.path.join(outdir, tag + e) for e in (".jsonl", ".done", ".ckpt")]
    if os.path.exists(done):
        return tag, 0.0
    pos_b, l_b, L0, start = net.pos.copy(), net.rest_length.copy(), None, 0
    V = np.linalg.qr(np.random.default_rng(12345).normal(size=(nb, K)))[0]
    if os.path.exists(ck):
        st = pickle.load(open(ck, "rb"))
        pos_b, l_b, V, start, L0 = st["pos"], st["l0"], st["V"], st["cycle"], st["L0"]
        keep = [l for l in open(path) if json.loads(l)["cycle"] < start] if os.path.exists(path) else []
        open(path, "w").writelines(keep)
        print(f"[{tag}] resumed at cycle {start}", flush=True)
    else:
        open(path, "w").close()
    t0 = time.time()
    with open(path, "a") as out:
        for c in range(start, cycles + 1):
            if c % 50 == 0 and c > start and c < cycles:
                pickle.dump(dict(pos=pos_b, l0=l_b, V=V, cycle=c, L0=L0), open(ck + ".tmp", "wb")); os.replace(ck + ".tmp", ck)
            if c % every == 0 or c == cycles:
                net.pos, net.rest_length = pos_b.copy(), l_b.copy()
                base, dL, ls, dls = ct.measure(net, source, targets)
                if L0 is None:
                    L0 = base.copy()
                lam_S, _ = ct.rs.lam_max_S(net, ct.minimize)
                rec = ct.make_record(seed, c, phases, base, dL, ls, dls, L0, net,
                                     dict(kind="meas", lam_max_S=lam_S, sigma=sigma, eps=eps, gamma=gamma, n_targets=nT, label=label))
                out.write(json.dumps(rec) + "\n"); out.flush()
                if c % (5 * every) == 0 or c == cycles:
                    print(f"[{tag}] [{(time.time() - t0) / 60:6.1f} min] cycle {c} E {rec['error']:.4g} E0 {rec['error_fixed_norm']:.4g} "
                          f"min l/l0 {float((base / L0).min()):.3g} lam_max(S) {lam_S:.4g}", flush=True)
            if c < cycles:
                pos_n, l_n = phi(net, source, targets, phases, eps, gamma, nsub, pos_b, l_b)
                W = np.empty((nb, K))
                for k in range(K):
                    _, lk = phi(net, source, targets, phases, eps, gamma, nsub, pos_b, l_b + delta * V[:, k])
                    W[:, k] = (lk - l_n) / delta
                Q, R = np.linalg.qr(W)
                logR = np.log(np.maximum(np.abs(np.diag(R)), 1e-300))
                out.write(json.dumps(dict(kind="step", cycle=c, logR=logR.tolist(), dl0=float(np.linalg.norm(l_n - l_b)))) + "\n")
                V, pos_b, l_b = Q, pos_n, l_n
    open(done, "w").write("done\n")
    if os.path.exists(ck):
        os.remove(ck)
    return tag, (time.time() - t0) / 60


def delta_check(label, quick, sigma, after):
    ct, net, source, targets, phases, seed, nT, eps, gamma, cycles, every = setup(label, quick, sigma)
    nsub = ct.NSUB_DEFAULT
    pos, l0 = net.pos.copy(), net.rest_length.copy()
    nb = len(l0)
    v = np.random.default_rng(7).normal(size=nb); v /= np.linalg.norm(v)
    deltas = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    for stage in ("untrained network", f"after {after} cycles"):
        if stage != "untrained network":
            for _ in range(after):
                pos, l0 = phi(net, source, targets, phases, eps, gamma, nsub, pos, l0)
        pn, ln = phi(net, source, targets, phases, eps, gamma, nsub, pos, l0)
        res = {}
        for d in deltas:
            _, lk = phi(net, source, targets, phases, eps, gamma, nsub, pos, l0 + d * v)
            res[d] = (lk - ln) / d
        print(f"\n{stage}:   |J v| for one fixed random unit vector v, forward differences")
        print(f"{'delta':>8} {'|J v|':>12} {'rel. change vs next smaller delta':>36}")
        for i, d in enumerate(deltas):
            rel = np.linalg.norm(res[d] - res[deltas[i + 1]]) / np.linalg.norm(res[d]) if i + 1 < len(deltas) else float("nan")
            print(f"{d:>8.0e} {np.linalg.norm(res[d]):>12.8f} {rel:>36.3e}")
        r1 = np.linalg.norm(res[1e-3] - res[1e-4]) / np.linalg.norm(res[1e-4])
        r2 = np.linalg.norm(res[1e-4] - res[1e-5]) / np.linalg.norm(res[1e-4])
        print("delta = 1e-4:", "OK (changes by less than 1e-3 when delta is 10x larger or 10x smaller)" if max(r1, r2) < 1e-3 else
              f"WARNING: |J v| changes by {max(r1, r2):.2e} between neighbouring deltas; send this table before running the long job")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", default="1000005,200005,700000")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--delta", type=float, default=1e-4)
    ap.add_argument("--sigma", type=float, default=0.0, help="optional excluded-volume range (default 0 = point nodes)")
    ap.add_argument("--cycles", type=int, default=None)
    ap.add_argument("--every", type=int, default=None)
    ap.add_argument("--n-workers", type=int, default=3)
    ap.add_argument("--outdir", default="results_cyc")
    ap.add_argument("--delta-check", action="store_true")
    ap.add_argument("--check-after", type=int, default=100)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    from run_excluded_volume import NETS
    nets = [n.strip() for n in a.nets.split(",") if n.strip()]
    for n in nets:
        if n not in NETS:
            sys.exit(f"unknown net {n}; choose from {list(NETS)}")
    if a.delta_check:
        delta_check(nets[0], a.quick, a.sigma, 3 if a.quick else a.check_after)
        return
    if a.quick:
        nets, a.K, a.outdir = nets[:2], 2, "results_cyc_quick"
        print("QUICK SMOKE TEST (N=40, 20 cycles, K=2) -- not real data.")
    nw = max(1, min(a.n_workers, 8, len(nets)))
    jobs = [(n, a.K, a.delta, a.sigma, a.outdir, a.quick, a.cycles, a.every) for n in nets]
    print(f"{len(jobs)} networks on {nw} workers, K={a.K}, delta={a.delta:g} -> {a.outdir}/", flush=True)
    t0 = time.time()
    with mp.Pool(nw) as pool:
        for k, (tag, mins) in enumerate(pool.imap_unordered(run_net, jobs), 1):
            print(f"== finished {tag} in {mins:.1f} min   ({k}/{len(jobs)}, total {(time.time() - t0) / 60:.1f} min)", flush=True)
    print("All done. Next: python analyze_cycle_lyapunov.py" + (" --dir results_cyc_quick" if a.quick else ""))


if __name__ == "__main__":
    main()
