"""
run_demo.py

A small, fast (a few minutes) run of the full pipeline, meant to prove the
machinery works and show what real output looks like -- NOT to make any
scientific claim. See README.md for exactly what would need to scale up
(N, cycle budget, grid resolution, number of realizations) for a result you
could put in a paper, and for an important parameter-sensitivity finding
from calibration (below) that you should resolve before trusting a big run.
"""
import numpy as np
import json

from spring_network import SpringNetwork
from training import train_one_cycle, measure_error_and_order_parameter, lowest_modes_with_overlap
from phase_diagram import run_phase_diagram, aggregate_phase_diagram


def calibration_note():
    print(__doc__)
    print(
        "CALIBRATION FINDING (found while building this, not in the original\n"
        "paper's stated protocol): with plasticity_rate and eps_age both on the\n"
        "more aggressive end (e.g. 0.02-0.05 and 0.06-0.08), error initially\n"
        "drops but then REVERSES and climbs back up after tens to a couple\n"
        "hundred cycles, on multiple realizations tried during development.\n"
        "With more conservative values (plasticity_rate ~0.01-0.015,\n"
        "eps_age ~0.03-0.05) descent is monotonic but slow. This smells like\n"
        "the quasistatic assumption (aging slow compared to elastic\n"
        "relaxation) being violated at the more aggressive settings, but I\n"
        "have not nailed down the exact threshold analytically, only\n"
        "empirically. Treat any specific (eps_age, plasticity_rate) pair as\n"
        "unverified until you've confirmed monotonic convergence at the N and\n"
        "cycle budget you actually intend to use -- don't trust the defaults\n"
        "here beyond this demo's tiny scale.\n"
    )


def demo_single_training_run():
    print("\n=== 1) Single training run, conservative/stable settings ===")
    rng = np.random.default_rng(123)
    net = SpringNetwork.random_diluted_delaunay(N=50, target_Z=4.8, rng=rng)
    source, targets, phases = (0, 1), [(10, 11)], np.array([1.0])

    for cycle in range(120):
        if cycle % 20 == 0:
            e, c = measure_error_and_order_parameter(net, source, targets, phases)
            print(f"  cycle {cycle:4d}  error={e:.4f}  converged={c}")
        train_one_cycle(net, source, targets, phases, eps_age=0.04, plasticity_rate=0.012, n_substeps=8)
    e, c = measure_error_and_order_parameter(net, source, targets, phases)
    print(f"  final   error={e:.4f}  converged={c}")


def demo_mode_tracking():
    print("\n=== 2) NOVEL: separate omega_trained / omega_next tracking ===")
    rng = np.random.default_rng(7)
    net = SpringNetwork.random_diluted_delaunay(N=50, target_Z=4.3, rng=rng)
    source, targets, phases = (0, 1), [(15, 16)], np.array([1.0])
    for _ in range(30):
        train_one_cycle(net, source, targets, phases, eps_age=0.05, plasticity_rate=0.02, n_substeps=8)
    info = lowest_modes_with_overlap(net, source, n_modes=6)
    print(f"  omega_trained    = {info['omega_trained']:.4f}")
    print(f"  omega_next       = {info['omega_next']:.4f}")
    print(f"  overlap_trained  = {info['overlap_trained']:.4f}  "
          f"(1.0 = one mode fully carries the trained response; well below 1 "
          f"means the original paper's implicit single-mode story is incomplete "
          f"for this network)")
    print(f"  rank_of_trained  = {info['rank_of_trained']}  (0 = lowest nonzero "
          f"mode, i.e. matches the paper's assumption; >0 would contradict it)")


def demo_tiny_phase_diagram():
    print("\n=== 3) NOVEL: tiny illustrative (Delta, dZ) grid ===")
    print("  (real grid: larger N, many more cycles, more realizations -- see README)")
    results = run_phase_diagram(
        N=35, dZ_values=[0.3, 0.8], delta_values=[0.06, 0.17],
        n_realizations=3, cycles=25, seed=2026,
        eps_age=0.04, plasticity_rate=0.012, n_substeps=6, measure_every=25,
    )
    rows = aggregate_phase_diagram(results)
    print(f"  {'dZ':>5} {'delta':>7} {'n':>3} {'err_mean':>9} {'err_sem':>8} "
          f"{'conv_frac':>9} {'w_trained':>10} {'w_next':>8} {'overlap':>8}")
    for r in rows:
        print(f"  {r['dZ']:5.2f} {r['delta']:7.3f} {r['n_realizations']:3d} "
              f"{r['error_mean']:9.4f} {r['error_sem']:8.4f} "
              f"{r['converged_fraction']:9.3f} {r['omega_trained_mean']:10.4f} "
              f"{r['omega_next_mean']:8.4f} {r['overlap_trained_mean']:8.4f}")
    with open("demo_phase_diagram_output.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("  (raw aggregated rows written to demo_phase_diagram_output.json)")


if __name__ == "__main__":
    calibration_note()
    demo_single_training_run()
    demo_mode_tracking()
    demo_tiny_phase_diagram()
    print("\nDone. This demo is deliberately tiny (small N, few cycles, few grid "
          "points) so it finishes in a couple of minutes -- it is a correctness "
          "check, not a result. See README.md for what to change to run the real thing.")
