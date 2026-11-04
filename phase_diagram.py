"""
phase_diagram.py

NOVEL (this whole file). The original paper sweeps Delta at only two values
of dZ, extrapolates Delta_c from a fit on the below-Delta_c side only, and
never separately reports omega_1 vs omega_2. This module instead:

  1. Sweeps a full (Delta, dZ) GRID, not two slices, so the non-monotonicity
     in Delta seen at large dZ (their own Fig. 2(d)) can actually be mapped
     as a function of dZ instead of asserted from one example.
  2. Runs MULTIPLE independent network realizations per grid point and
     reports mean +/- standard error, instead of a single realization per
     point with a "guide to the eye" line.
  3. Runs training for a fixed, generous cycle budget at EVERY point
     (including deep above the naively expected Delta_c) and records the
     measured converged-fraction as an actual order parameter, rather than
     only fitting alpha(Delta) from below and extrapolating where it hits
     zero.
  4. Records omega_trained and omega_next SEPARATELY (see training.py) at
     the end of the cycle budget for every point, plus the overlap
     diagnostic, so the mechanism itself (not just its consequence) can be
     checked across the whole grid.

This file is the actual deliverable of the project. spring_network.py and
training.py exist to make this one honest.
"""

import itertools
import numpy as np

from spring_network import SpringNetwork
from training import train_one_cycle, measure_error_and_order_parameter, lowest_modes_with_overlap


def run_single_point(N, dZ, delta, n_targets_total, cycles, rng,
                      eps_age=0.05, plasticity_rate=0.02, n_substeps=8,
                      measure_every=10, converged_threshold=0.1):
    """
    Train one network realization at coordination dZ (=Z-4) with
    Delta = n_targets/N fixed by choosing n_targets = round(delta * N),
    for `cycles` training cycles, measuring error periodically.

    Returns a dict with the full error trajectory and the final mode-overlap
    diagnostics, for one realization. Aggregation across realizations and
    across the (Delta, dZ) grid happens in `run_phase_diagram`.
    """
    net = SpringNetwork.random_diluted_delaunay(N=N, target_Z=4.0 + dZ, rng=rng)

    n_targets = max(1, int(round(delta * N)))
    all_nodes = rng.permutation(N)
    source = tuple(all_nodes[:2])
    remaining = all_nodes[2:]
    if len(remaining) < 2 * n_targets:
        n_targets = len(remaining) // 2
    targets = [tuple(remaining[2 * t: 2 * t + 2]) for t in range(n_targets)]
    phases = rng.choice([-1.0, 1.0], size=n_targets)

    trajectory = []
    for cycle in range(cycles):
        if cycle % measure_every == 0:
            err, conv = measure_error_and_order_parameter(
                net, source, targets, phases, converged_threshold=converged_threshold)
            trajectory.append((cycle, err, conv))
        train_one_cycle(net, source, targets, phases, eps_age=eps_age,
                         plasticity_rate=plasticity_rate, n_substeps=n_substeps)

    err_final, conv_final = measure_error_and_order_parameter(
        net, source, targets, phases, converged_threshold=converged_threshold)
    trajectory.append((cycles, err_final, conv_final))

    modes = lowest_modes_with_overlap(net, source, n_modes=6)

    return {
        "N": N, "dZ": dZ, "delta": delta, "n_targets": n_targets,
        "trajectory": trajectory,
        "error_final": err_final,
        "converged_final": conv_final,
        **modes,
    }


def run_phase_diagram(N, dZ_values, delta_values, n_realizations, cycles,
                       seed=0, **kwargs):
    """
    Full grid sweep. Returns a list of per-realization result dicts (as from
    `run_single_point`); aggregate with `aggregate_phase_diagram`.

    This is the expensive part: cost scales as
        len(dZ_values) * len(delta_values) * n_realizations * cycles
    energy minimizations (times n_substeps each). For anything beyond a
    quick correctness check, this needs to run outside an interactive
    session -- see run_demo.py for a small, fast, illustrative grid, and the
    module docstring in README.md for what a real production grid costs.
    """
    master_rng = np.random.default_rng(seed)
    results = []
    combos = list(itertools.product(dZ_values, delta_values))
    for dZ, delta in combos:
        for r in range(n_realizations):
            rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
            res = run_single_point(N, dZ, delta, n_targets_total=None,
                                    cycles=cycles, rng=rng, **kwargs)
            res["realization"] = r
            results.append(res)
    return results


def aggregate_phase_diagram(results):
    """
    Collapse per-realization results into per-(dZ, delta) statistics:
    mean +/- SEM of final error, converged fraction (the order parameter,
    defined and measured identically on both sides of Delta_c), and mean
    omega_trained / omega_next / overlap.
    """
    import collections
    grouped = collections.defaultdict(list)
    for r in results:
        grouped[(r["dZ"], r["delta"])].append(r)

    rows = []
    for (dZ, delta), group in sorted(grouped.items()):
        errs = np.array([g["error_final"] for g in group])
        conv = np.array([g["converged_final"] for g in group])
        w1 = np.array([g["omega_trained"] for g in group])
        w2 = np.array([g["omega_next"] for g in group])
        ov = np.array([g["overlap_trained"] for g in group])
        n = len(group)
        rows.append({
            "dZ": dZ, "delta": delta, "n_realizations": n,
            "error_mean": float(errs.mean()),
            "error_sem": float(errs.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
            "converged_fraction": float(conv.mean()),  # NOVEL order parameter
            "omega_trained_mean": float(w1.mean()),
            "omega_next_mean": float(w2.mean()),
            "overlap_trained_mean": float(ov.mean()),
        })
    return rows
