"""
training.py

Quasistatic training and measurement protocol for SpringNetwork objects.

REPRODUCTION
------------
`minimize`, `train_one_cycle`, and `measure_error` reproduce the published
protocol: drive a source pair and a set of target pairs through a strain
cycle (0 -> +eps_age -> 0 -> -eps_age -> 0) with each target's phase chosen
once (in-phase / out-of-phase with the source), quasistatically relax the
network at each sub-step, and evolve every real bond's rest length toward
relaxing its own current tension (Eq. 1 of Bhaumik & Hexner 2022):
    d(ell_0)/dt = plasticity_rate * k * (ell - ell_0)
Error is measured by driving the source ALONE (targets left free) and
comparing the resulting target-pair strain to the desired sign.

NOVELTY (flagged inline with "# NOVEL:")
-----------------------------------------
1. `lowest_modes_with_overlap` computes the low end of the vibrational
   spectrum of the *untrained* (ghost-free) network at each measurement
   point and identifies which mode is actually "the trained one" by its
   overlap with the true displacement pattern induced by the source, rather
   than assuming (as the original paper does) that it is always simply the
   lowest nonzero mode. omega_1 (trained-mode frequency) and omega_2 (next
   lowest, "spurious") are returned SEPARATELY, never pre-combined into a
   ratio, and their identities are re-checked every time.
2. `measure_error_and_order_parameter` reports the error as a continuous
   quantity together with a converged/not-converged flag, at every Delta,
   not only in the regime approached from below Delta_c, so that the
   Delta > Delta_c side has an actual measured order parameter rather than
   an extrapolation.
"""

import numpy as np
from scipy.optimize import minimize as scipy_minimize


# ----------------------------------------------------------------------
# Quasistatic relaxation
# ----------------------------------------------------------------------
def minimize(net, x0=None, tol=1e-10, maxiter=2000):
    """Relax net.pos to a local energy minimum given current bonds + ghosts.
    Returns the flattened equilibrium position vector; also updates net.pos."""
    if x0 is None:
        x0 = net.pos.reshape(-1)

    res = scipy_minimize(
        fun=lambda x: net.energy(x.reshape(net.N, 2)),
        x0=x0,
        jac=lambda x: net.gradient(x.reshape(net.N, 2)),
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-16, "gtol": tol},
    )
    net.pos = res.x.reshape(net.N, 2)
    return res.x


# ----------------------------------------------------------------------
# REPRODUCTION: one training cycle
# ----------------------------------------------------------------------
def train_one_cycle(net, source, targets, target_phases, eps_age,
                     k_ghost=50.0, n_substeps=12, plasticity_rate=0.02):
    """
    Drive source and all targets through one full strain cycle
    (0 -> +eps_age -> 0 -> -eps_age -> 0), relaxing and plastically updating
    rest lengths at each sub-step (Eq. 1). `target_phases` is a length-len(targets)
    array of +1/-1 fixed once at setup (in-phase / out-of-phase).
    """
    # Fix the "zero-strain" reference length for source and each target ONCE,
    # at the undriven equilibrium before this cycle starts. Recomputing it
    # fresh at every substep (an earlier bug in this file) lets the
    # reference drift as the network relaxes and plastically evolves within
    # the cycle, which corrupts the intended triangle-wave strain and was
    # empirically found to make training diverge instead of converge.
    net.clear_ghosts()
    minimize(net)
    l0_source = _driven_length(net, source)
    l0_targets = [_driven_length(net, t) for t in targets]

    schedule = _cycle_schedule(n_substeps)
    for s in schedule:
        net.set_ghost("source", source[0], source[1], k_ghost,
                      rest_length=l0_source * (1 + s * eps_age))
        for t_idx, (ta, tb) in enumerate(targets):
            phase = target_phases[t_idx]
            net.set_ghost(f"target_{t_idx}", ta, tb, k_ghost,
                          rest_length=l0_targets[t_idx] * (1 + phase * s * eps_age))
        minimize(net)
        _plastic_update(net, plasticity_rate)
    net.clear_ghosts()


def _plastic_update(net, plasticity_rate):
    """Eq. 1: rest length relaxes toward current length, rate ~ k * tension."""
    _, length = net.bond_vectors(net.pos)
    net.rest_length += plasticity_rate * net.k * (length - net.rest_length)


def _cycle_schedule(n_substeps):
    """Discretized 0 -> +1 -> 0 -> -1 -> 0 strain-amplitude schedule."""
    up = np.linspace(0, 1, n_substeps // 4 + 1)[1:]
    down = np.linspace(1, 0, n_substeps // 4 + 1)[1:]
    dn = np.linspace(0, -1, n_substeps // 4 + 1)[1:]
    back = np.linspace(-1, 0, n_substeps // 4 + 1)[1:]
    return np.concatenate([up, down, dn, back])


def _driven_length(net, pair):
    from spring_network import _minimum_image_sep
    sep = _minimum_image_sep(net.pos[pair[0]], net.pos[pair[1]], net.L)
    return np.linalg.norm(sep)


# ----------------------------------------------------------------------
# REPRODUCTION (+ NOVEL order-parameter framing): measuring the response
# ----------------------------------------------------------------------
def measure_error_and_order_parameter(net, source, targets, target_phases,
                                       eps_measure=1e-3, k_ghost=50.0,
                                       converged_threshold=0.1):
    """
    Drive the source ALONE with a small test strain, leave targets free,
    measure the resulting fractional strain at each target, and compare to
    the desired sign (target_phases). Returns:
        error : mean |actual_strain/eps_measure - desired_sign| over targets
                (0 = perfect, ~1 = no better than a null response, this is
                the delta_epsilon quantity in the original paper)
        converged : bool, error < converged_threshold
    # NOVEL: unlike the original paper, this is called identically on both
    # sides of Delta_c, at a fixed, generous cycle budget, so "converged"
    # is an actual measured order parameter across the whole Delta range,
    # not an extrapolation from the below-Delta_c side only.
    """
    net.clear_ghosts()
    minimize(net)  # relax to the true, undriven equilibrium first -> baseline
    baseline_lengths = np.array([_driven_length(net, t) for t in targets])
    x_ref = net.pos.reshape(-1).copy()

    l0 = _driven_length(net, source)
    net.set_ghost("source", source[0], source[1], k_ghost, rest_length=l0 * (1 + eps_measure))
    minimize(net)
    driven_lengths = np.array([_driven_length(net, t) for t in targets])

    net.clear_ghosts()
    net.pos = x_ref.reshape(net.N, 2)  # restore undriven state; caller keeps training from here

    actual_strain = (driven_lengths - baseline_lengths) / (baseline_lengths * eps_measure)
    errs = np.abs(actual_strain - target_phases)
    net.clear_ghosts()

    error = float(np.mean(errs))
    return error, error < converged_threshold


# ----------------------------------------------------------------------
# NOVEL: separate omega_1 / omega_2 tracking by overlap, not by rank
# ----------------------------------------------------------------------
def lowest_modes_with_overlap(net, source, n_modes=8, zero_mode_cutoff=1e-8):
    """
    Diagonalize the ghost-free Hessian at the current (trained) equilibrium,
    discard near-zero (translational) modes, and identify which of the
    lowest `n_modes` real modes has the largest overlap with the actual
    displacement field induced by straining the source pair alone.

    Returns dict with:
        omega_trained : frequency (sqrt eigenvalue) of the best-overlap mode
        omega_next    : frequency of the next-lowest mode that is NOT the
                        trained one (by index, after removing zero modes)
        overlap_trained : |<source displacement, trained eigenvector>|
                           (normalized, in [0, 1]) -- if this is small even
                           for the "best" mode, no single mode is really
                           carrying the trained response and omega_trained
                           should be treated with suspicion, not silently
                           accepted the way rank-1 always was.
        rank_of_trained : index (0 = lowest nonzero mode) of the trained
                           mode. The original paper's implicit assumption is
                           that this is always 0; we check it explicitly.
    """
    net.clear_ghosts()
    minimize(net)
    H = net.hessian(net.pos, include_ghosts=False)
    N = net.num_dof()
    eigvals, eigvecs = np.linalg.eigh(H)

    nonzero = eigvals > zero_mode_cutoff
    eigvals = eigvals[nonzero]
    eigvecs = eigvecs[:, nonzero]

    order = np.argsort(eigvals)
    eigvals = eigvals[order][:n_modes]
    eigvecs = eigvecs[:, order][:, :n_modes]

    # displacement induced by straining the source pair by a small test
    # amount, relative to the ghost-free equilibrium (this is the "trained
    # direction" against which every low mode is checked for overlap)
    x_ref = net.pos.reshape(-1).copy()
    net.set_ghost("source", source[0], source[1], k_ghost=50.0,
                  rest_length=_driven_length(net, source) * 1.001)
    minimize(net)
    disp = net.pos.reshape(-1) - x_ref
    disp_norm = disp / (np.linalg.norm(disp) + 1e-14)
    net.clear_ghosts()
    net.pos = x_ref.reshape(net.N, 2)

    overlaps = np.abs(eigvecs.T @ disp_norm)
    trained_idx = int(np.argmax(overlaps))
    remaining = [i for i in range(len(eigvals)) if i != trained_idx]
    next_idx = remaining[0] if remaining else trained_idx

    return {
        "omega_trained": float(np.sqrt(eigvals[trained_idx])),
        "omega_next": float(np.sqrt(eigvals[next_idx])),
        "overlap_trained": float(overlaps[trained_idx]),
        "rank_of_trained": trained_idx,
        "all_omegas": np.sqrt(eigvals).tolist(),
        "all_overlaps": overlaps.tolist(),
    }
