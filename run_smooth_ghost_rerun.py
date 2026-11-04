"""
run_smooth_ghost_rerun.py

Reruns the SAME realizations underlying Table 1 (same seeds, same networks,
same eps_age / plasticity_rate) with ONE change: the driving "ghost" springs
use a periodic image fixed at the start of each strain cycle, instead of a
minimum-image convention re-evaluated at every energy call.

Why: the original ghost energy uses a minimum-image separation between
arbitrary node pairs. When a pair's separation component reaches L/2 the two
images are equally close, the energy has a kink, and the analytic gradient is
inconsistent there (finite-difference slope +0.128 vs analytic -0.192 was
measured on seed 1000005). L-BFGS-B then fails with ABNORMAL line-search
termination. With the fixed-image version, seed 1000005's late blow-up
(error 17.7 near cycle 360) disappears (max error 1.41 through cycle 440) and
solver failures drop from ~8.5% to ~0.7%. Seed 200005's early spike is
unchanged (identical trajectory), so some divergences survive.

This script tells you, for each realization, whether its divergence survives.
It also records the Hessian-based quantity lambda_max(S), S = B H^+ B^T K
(response of equilibrium bond lengths to rest lengths; the linearised
plasticity step has Jacobian (1-gamma) I + gamma S), at every measurement
point, to test a stability criterion on the same data.

Usage (same folder as spring_network.py, training.py, run_solver_failure_check.py):
    python run_smooth_ghost_rerun.py                 # 20 divergent + 20 stable
    python run_smooth_ghost_rerun.py --n-stable 81   # all 101
    python run_smooth_ghost_rerun.py --n-workers 8
    python run_smooth_ghost_rerun.py --quick         # smoke test

Appends to smooth_ghost_results.jsonl as jobs complete.
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import time

import numpy as np

from run_solver_failure_check import REALIZATIONS  # same 101 (n_targets, seed, label)

N, DZ = 100, 0.5
EPS_AGE, PLASTICITY_RATE, N_SUBSTEPS = 0.04, 0.012, 8
MEASURE_EVERY, CYCLES = 20, 800
OUT_PATH = "smooth_ghost_results.jsonl"


def install_smooth_ghosts():
    from spring_network import SpringNetwork

    def set_ghost(self, name, i, j, k_ghost, rest_length):
        if name in self.ghosts:
            self.ghosts[name]["rest_length"] = rest_length
        else:
            d = self.pos[j] - self.pos[i]
            self.ghosts[name] = {"pair": (i, j), "k": k_ghost, "rest_length": rest_length,
                                 "image": np.zeros(2, int), "shift": -np.round(d / self.L)}

    def energy(self, pos):
        sep, length = self.bond_vectors(pos)
        e = 0.5 * np.sum(self.k * (length - self.rest_length) ** 2)
        for g in self.ghosts.values():
            sg = pos[g["pair"][1]] - pos[g["pair"][0]] + g["shift"] * self.L
            e += 0.5 * g["k"] * (np.linalg.norm(sg) - g["rest_length"]) ** 2
        return e

    def gradient(self, pos):
        grad = np.zeros((self.N, 2))
        sep, length = self.bond_vectors(pos)
        n = sep / length[:, None]
        f_mag = self.k * (length - self.rest_length)
        i, j = self.bonds[:, 0], self.bonds[:, 1]
        np.add.at(grad, i, -f_mag[:, None] * n)
        np.add.at(grad, j, f_mag[:, None] * n)
        for g in self.ghosts.values():
            gi, gj = g["pair"]
            sg = pos[gj] - pos[gi] + g["shift"] * self.L
            lg = np.linalg.norm(sg)
            ng = sg / lg
            f = g["k"] * (lg - g["rest_length"])
            grad[gi] += -f * ng
            grad[gj] += f * ng
        return grad.reshape(-1)

    SpringNetwork.set_ghost = set_ghost
    SpringNetwork.energy = energy
    SpringNetwork.gradient = gradient


def lam_max_S(net, minimize):
    net.clear_ghosts()
    minimize(net)
    sep, ln = net.bond_vectors(net.pos)
    n = sep / ln[:, None]
    nb = len(net.bonds)
    B = np.zeros((nb, 2 * net.N))
    for b, (i, j) in enumerate(net.bonds):
        B[b, 2 * i:2 * i + 2] = -n[b]
        B[b, 2 * j:2 * j + 2] = n[b]
    H = net.hessian(net.pos, include_ghosts=False)
    w = np.linalg.eigvalsh(H)
    Hp = np.linalg.pinv(H, rcond=1e-9)
    sk = np.sqrt(net.k)
    M = (sk[:, None] * B) @ Hp @ (B.T * sk[None, :])
    return float(np.linalg.eigvalsh((M + M.T) / 2).max()), float(w[w > 1e-8].min())


def run_one(job):
    import training
    from scipy.optimize import minimize as real_min
    from training import train_one_cycle, measure_error_and_order_parameter, lowest_modes_with_overlap
    from spring_network import SpringNetwork

    install_smooth_ghosts()
    n_targets, seed, label, cfg = job
    log = []

    def logging_minimize(fun, x0, jac=None, method=None, options=None):
        res = real_min(fun=fun, x0=x0, jac=jac, method=method, options=options)
        log.append(bool(res.success))
        return res

    training.scipy_minimize = logging_minimize
    rng = np.random.default_rng(seed)
    net = SpringNetwork.random_diluted_delaunay(N=cfg["N"], target_Z=4.0 + cfg["dZ"], rng=rng)
    nodes = rng.permutation(cfg["N"])
    source = tuple(nodes[:2])
    targets = [tuple(nodes[2 + 2 * t: 4 + 2 * t]) for t in range(n_targets)]
    phases = rng.choice([-1.0, 1.0], size=n_targets)

    traj, t0 = [], time.time()
    for cycle in range(cfg["cycles"] + 1):
        if cycle % cfg["measure_every"] == 0:
            err, conv = measure_error_and_order_parameter(net, source, targets, phases)
            modes = lowest_modes_with_overlap(net, source, n_modes=6)
            lam, w2 = lam_max_S(net, training.minimize)
            traj.append({"cycle": cycle, "error": err, "omega_trained": modes["omega_trained"],
                         "rank": modes["rank_of_trained"], "lam_max": lam, "omega2_min": w2})
        if cycle < cfg["cycles"]:
            train_one_cycle(net, source, targets, phases, eps_age=cfg["eps_age"],
                            plasticity_rate=cfg["plasticity_rate"], n_substeps=cfg["n_substeps"])
    errs = np.array([p["error"] for p in traj])
    return {"n_targets": n_targets, "seed": seed, "label_original": "blowup" if label else "stable",
            "wall_time_s": time.time() - t0, "n_minimize_calls": len(log),
            "n_minimize_failures": int(sum(1 for ok in log if not ok)),
            "peak_error": float(errs.max()), "init_error": float(errs[0]),
            "peak_over_init": float(errs.max() / errs[0]), "final_error": float(errs[-1]),
            "trajectory": traj}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--n-stable", type=int, default=20)
    ap.add_argument("--select-seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    div = [r for r in REALIZATIONS if r[2]]
    stab = [r for r in REALIZATIONS if not r[2]]
    chosen = random.Random(args.select_seed).sample(stab, min(args.n_stable, len(stab)))
    sel = div + chosen
    cfg = dict(N=N, dZ=DZ, cycles=CYCLES, eps_age=EPS_AGE, plasticity_rate=PLASTICITY_RATE,
               n_substeps=N_SUBSTEPS, measure_every=MEASURE_EVERY)
    if args.quick:
        cfg.update(N=40, cycles=60)
        sel = sel[:2]
        print("QUICK SMOKE TEST -- not real data.")
    jobs = [(nt, s, lab, cfg) for nt, s, lab in sel]
    nw = args.n_workers or os.cpu_count()
    print(f"{len(jobs)} realizations across {nw} workers, N={cfg['N']}, cycles={cfg['cycles']}")
    t0, done = time.time(), 0
    with mp.Pool(nw) as pool, open(OUT_PATH, "a") as f:
        for res in pool.imap_unordered(run_one, jobs):
            f.write(json.dumps(res) + "\n")
            f.flush()
            done += 1
            print(f"[{(time.time()-t0)/60:6.1f} min] {done}/{len(jobs)} {res['label_original']} "
                  f"n_targets={res['n_targets']} peak/init={res['peak_over_init']:.2f} "
                  f"fail={res['n_minimize_failures']}/{res['n_minimize_calls']}", flush=True)
    print(f"All done in {(time.time()-t0)/60:.1f} min -> {OUT_PATH}")


if __name__ == "__main__":
    main()