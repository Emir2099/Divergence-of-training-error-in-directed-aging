"""
cycle_tools.py -- shared helpers for the four compute scripts (excluded volume, ensemble re-recording,
cycle-map growth rates, onset map).  Keep this file in the SAME folder as spring_network.py, training.py,
run_smooth_ghost_rerun.py and run_solver_failure_check.py.

What it provides
  build(seed, n_targets)            the same network / source / targets / phases as run_collapse_check.py
  ExcludedVolume, attach_excluded_volume(net, sigma, krep)
                                    optional short-range repulsion between ALL node pairs (see below)
  measure(net, source, targets)     readout only: base target lengths, absolute response of the targets to a
                                    1e-3 source drive, and the same for the source pair.  Aging never uses it.
  run_trajectory(...)               train for a number of cycles, record the readouts every `every` cycles,
                                    write <outdir>/<tag>.jsonl, checkpoint every 100 cycles, resume, write <tag>.done

Excluded volume
  E_rep = 0.5 * krep * (s_ij - r_ij)^2   for r_ij < s_ij,  summed over all node pairs (minimum image),
  s_ij = min(sigma, 0.9 * d_ij(0)),  d_ij(0) = distance of the pair in the initial (unstressed) network.
  So the initial network has no contact and its energy and forces are exactly those of the point-node model;
  a pair that starts closer than sigma/0.9 can only shrink to 90 % of its initial distance, any other pair to sigma.
  sigma = 0 attaches nothing (this is the original point-node model, bit-for-bit).
  Units: mean node spacing = 1 (box side sqrt(N)).  Bond stiffness = 1; krep = 1 by default.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import pickle
import time

import numpy as np

from training import train_one_cycle, minimize, _driven_length
from spring_network import SpringNetwork
import run_smooth_ghost_rerun as rs

rs.install_smooth_ghosts()          # ghost springs with a fixed periodic image (the corrected version used in the paper)

N_DEFAULT, DZ_DEFAULT, NSUB_DEFAULT = 100, 0.5, 8


# ----------------------------------------------------------------------------------------------- network
def build(seed, n_targets, N=N_DEFAULT, dZ=DZ_DEFAULT):
    """Same random-number sequence as run_collapse_check.py / run_smooth_ghost_rerun.py."""
    rng = np.random.default_rng(seed)
    net = SpringNetwork.random_diluted_delaunay(N=N, target_Z=4.0 + dZ, rng=rng)
    nodes = rng.permutation(N)
    source = tuple(int(i) for i in nodes[:2])
    targets = [tuple(int(i) for i in nodes[2 + 2 * t: 4 + 2 * t]) for t in range(n_targets)]
    phases = rng.choice([-1.0, 1.0], size=n_targets)
    return net, source, targets, phases


# ------------------------------------------------------------------------------------------ excluded volume
class ExcludedVolume:
    SKIN = 0.4      # neighbour-list skin (units of mean node spacing); only affects speed, not the physics

    def __init__(self, net, sigma, krep=1.0):
        self.sigma, self.krep, self.L = float(sigma), float(krep), net.L
        self.I, self.J = np.triu_indices(net.N, 1)
        self.d0 = self.dist(net.pos)                         # initial pair distances
        self.s = np.minimum(self.sigma, 0.9 * self.d0)       # contact distance of every pair
        self._ref = None

    def dist(self, pos):
        d = pos[self.J] - pos[self.I]
        d -= self.L * np.round(d / self.L)
        return np.hypot(d[:, 0], d[:, 1])

    def _active(self, pos):
        """Pairs that can be in contact: r_ij < s_ij + skin at a reference configuration; the list is rebuilt as soon as
        any node has moved more than skin/2 from the reference, so no contact is ever missed."""
        if self._ref is not None:
            d = pos - self._ref
            d -= self.L * np.round(d / self.L)
            if float(np.max(np.einsum("ij,ij->i", d, d))) < (0.5 * self.SKIN) ** 2:
                return self._ai, self._aj, self._as
        m = self.dist(pos) < self.s + self.SKIN
        self._ai, self._aj, self._as = self.I[m], self.J[m], self.s[m]
        self._ref = pos.copy()
        return self._ai, self._aj, self._as

    def energy(self, pos):
        ai, aj, s = self._active(pos)
        d = pos[aj] - pos[ai]
        d -= self.L * np.round(d / self.L)
        ov = np.maximum(s - np.hypot(d[:, 0], d[:, 1]), 0.0)
        return 0.5 * self.krep * float(np.sum(ov * ov))

    def gradient(self, pos):
        ai, aj, s = self._active(pos)
        d = pos[aj] - pos[ai]
        d -= self.L * np.round(d / self.L)
        r = np.hypot(d[:, 0], d[:, 1])
        ov = np.maximum(s - r, 0.0)
        g = np.zeros((pos.shape[0], 2))
        m = ov > 0
        if m.any():
            f = (self.krep * ov[m] / r[m])[:, None] * d[m]   # dE/dx_i = +f , dE/dx_j = -f
            np.add.at(g, ai[m], f)
            np.add.at(g, aj[m], -f)
        return g.reshape(-1)

    def stats(self, pos):
        r = self.dist(pos)
        ov = np.maximum(self.s - r, 0.0)
        return dict(min_pair_dist=float(r.min()), min_pair_ratio=float((r / self.d0).min()),
                    n_contacts=int((ov > 0).sum()), max_overlap=float(ov.max()),
                    e_rep=float(0.5 * self.krep * np.sum(ov * ov)))


def attach_excluded_volume(net, sigma, krep=1.0):
    """Call right after build(), before any training. sigma <= 0: nothing is attached (point nodes)."""
    ev = ExcludedVolume(net, sigma, krep)
    net.ev = ev
    if sigma > 0:
        cls = type(net)
        net.energy = lambda pos: cls.energy(net, pos) + ev.energy(pos)
        net.gradient = lambda pos: cls.gradient(net, pos) + ev.gradient(pos)
    return ev


# --------------------------------------------------------------------------------------------- readouts
def measure(net, source, targets, k_ghost=50.0, eps_measure=1e-3):
    """Readout only. Returns (base target lengths, dL targets, source length, dL source) for a source drive of
    relative size eps_measure at the undriven equilibrium.  Restores the undriven positions afterwards."""
    net.clear_ghosts()
    minimize(net)
    base = np.array([_driven_length(net, t) for t in targets])
    x_ref = net.pos.reshape(-1).copy()
    ls = _driven_length(net, source)
    net.set_ghost("source", source[0], source[1], k_ghost, rest_length=ls * (1 + eps_measure))
    minimize(net)
    drv = np.array([_driven_length(net, t) for t in targets])
    src_drv = _driven_length(net, source)
    net.clear_ghosts()
    net.pos = x_ref.reshape(net.N, 2)
    return base, drv - base, ls, src_drv - ls


def make_record(seed, cycle, phases, base, dL, ls, dls, L0, net=None, extra=None):
    err = float(np.mean(np.abs(dL / (base * 1e-3) - phases)))          # normalised by the CURRENT separation (the paper's E)
    err0 = float(np.mean(np.abs(dL / (L0 * 1e-3) - phases)))           # normalised by the INITIAL separation (E0)
    rec = dict(seed=seed, cycle=cycle, error=err, error_fixed_norm=err0,
               target_len=base.tolist(), target_dL=dL.tolist(), source_len=float(ls), source_dL=float(dls),
               gain=(dL / dls).tolist() if dls != 0 else None, phases=phases.tolist())
    if net is not None and getattr(net, "ev", None) is not None:
        rec.update(net.ev.stats(net.pos))
    if extra:
        rec.update(extra)
    return rec


# ------------------------------------------------------------------------------------------- one trajectory
def run_trajectory(tag, outdir, seed, nT, eps, gamma, cycles, every=20, sigma=0.0, krep=1.0,
                   N=N_DEFAULT, dZ=DZ_DEFAULT, nsub=NSUB_DEFAULT, ckpt_every=100, log=print, extra=None, stop=None):
    """Train one network and record the readouts. Resumable. Returns the path of the jsonl file.
    The ONLY thing that sigma changes is the energy used by the relaxation inside the training cycle and the readout.
    stop(cycles, errors) -> bool, optional: called after every measurement with the lists of measured cycles and errors
    (including those of a resumed run); returning True ends the run early (the run is then marked done)."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, tag + ".jsonl")
    done = os.path.join(outdir, tag + ".done")
    ck = os.path.join(outdir, tag + ".ckpt")
    if os.path.exists(done):
        log(f"[{tag}] already done")
        return path
    net, source, targets, phases = build(seed, nT, N, dZ)
    attach_excluded_volume(net, sigma, krep)
    start, L0, hist_c, hist_e = 0, None, [], []
    if os.path.exists(ck):
        st = pickle.load(open(ck, "rb"))
        net.pos, net.rest_length, start, L0 = st["pos"], st["l0"], st["cycle"], st["L0"]
        keep = [l for l in open(path)] if os.path.exists(path) else []
        keep = [l for l in keep if json.loads(l)["cycle"] < start]
        open(path, "w").writelines(keep)
        hist_c = [json.loads(l)["cycle"] for l in keep]
        hist_e = [json.loads(l)["error"] for l in keep]
        log(f"[{tag}] resumed at cycle {start}")
    else:
        open(path, "w").close()
    t0 = time.time()
    with open(path, "a") as out:
        for c in range(start, cycles + 1):
            if c % ckpt_every == 0 and c > start and c < cycles:
                pickle.dump(dict(pos=net.pos.copy(), l0=net.rest_length.copy(), cycle=c, L0=L0), open(ck + ".tmp", "wb"))
                os.replace(ck + ".tmp", ck)
            if c % every == 0 or c == cycles:
                base, dL, ls, dls = measure(net, source, targets)
                if L0 is None:
                    L0 = base.copy()
                rec = make_record(seed, c, phases, base, dL, ls, dls, L0, net,
                                  dict(sigma=sigma, krep=krep, eps=eps, gamma=gamma, n_targets=nT, **(extra or {})))
                out.write(json.dumps(rec) + "\n")
                out.flush()
                hist_c.append(c); hist_e.append(rec["error"])
                if stop is not None and stop(hist_c, hist_e):
                    log(f"[{tag}] stopped early at cycle {c} (stop rule)")
                    break
                if c % (5 * every) == 0 or c == cycles:
                    log(f"[{tag}] [{(time.time() - t0) / 60:6.1f} min] cycle {c} E {rec['error']:.4g} E0 {rec['error_fixed_norm']:.4g} "
                        f"min l_t/l_t(0) {float((base / L0).min()):.3g}", )
            if c < cycles:
                train_one_cycle(net, source, targets, phases, eps_age=eps, plasticity_rate=gamma, n_substeps=nsub)
    open(done, "w").write("done\n")
    if os.path.exists(ck):
        os.remove(ck)
    return path
