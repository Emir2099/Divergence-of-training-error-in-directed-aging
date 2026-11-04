"""
analyze_excluded_volume.py -- summary table, verdicts and figure for run_excluded_volume.py.
    python analyze_excluded_volume.py [--dir results_ev]
Writes <dir>/ev_summary.csv, <dir>/ev_summary.json, <dir>/ev_figure.png / .pdf and prints the table and a verdict per network.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, VERM, GREEN, ORANGE, SKY, PURP, GREY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7", "#8a8a8a"
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
SIGMA_COLOR = {0.0: VERM, 0.25: ORANGE, 0.5: GREEN, 1.0: BLUE}
EXC_FACTOR = 5.0          # an "excursion" of E: E_max > 5 x E at cycle 0 (the criterion used in the manuscript)
COLLAPSE_RATIO = 0.1      # a "collapse": a target separation below 0.1 of its initial value


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def summarise(recs):
    r0 = recs[0]
    L0 = np.array(r0["target_len"])
    sigma = r0["sigma"]
    cyc = np.array([r["cycle"] for r in recs])
    tl = np.array([r["target_len"] for r in recs])                       # (n_meas, nT)
    ratio = tl / L0
    E = np.array([r["error"] for r in recs]); E0 = np.array([r["error_fixed_norm"] for r in recs])
    gains = np.array([[g if g is not None else np.nan for g in r["gain"]] for r in recs])
    src = np.array([r["source_len"] for r in recs]) / r0["source_len"]
    floor = np.minimum(sigma, 0.9 * L0) / L0 if sigma > 0 else np.zeros_like(L0)    # ratio at which a target pair touches
    i_min = np.unravel_index(np.argmin(ratio), ratio.shape)
    out = dict(label=r0.get("label"), seed=r0["seed"], sigma=sigma, n_targets=r0["n_targets"], eps=r0["eps"], gamma=r0["gamma"],
               last_cycle=int(cyc[-1]), n_meas=len(recs),
               min_ratio=float(ratio.min()), min_ratio_cycle=int(cyc[i_min[0]]), min_ratio_target=int(i_min[1]),
               contact_floor_of_that_target=float(floor[i_min[1]]),
               at_contact=bool(sigma > 0 and np.any(ratio.min(axis=0) < 1.25 * floor)),
               min_source_ratio=float(src.min()),
               E_init=float(E[0]), E_max=float(E.max()), E_max_cycle=int(cyc[np.argmax(E)]), E_max_over_init=float(E.max() / E[0]),
               E0_min=float(E0.min()), E0_max=float(E0.max()), E0_max_cycle=int(cyc[np.argmax(E0)]),
               abs_gain_max=float(np.nanmax(np.abs(gains))),
               n_contacts_max=int(max(r.get("n_contacts", 0) for r in recs)),
               max_overlap_max=float(max(r.get("max_overlap", 0.0) for r in recs)),
               e_rep_max=float(max(r.get("e_rep", 0.0) for r in recs)),
               min_pair_ratio_any_pair=float(min(r.get("min_pair_ratio", np.nan) for r in recs)),
               min_pair_dist_any_pair=float(min(r.get("min_pair_dist", np.nan) for r in recs)))
    out["collapsed"] = out["min_ratio"] < COLLAPSE_RATIO
    out["E_excursion"] = out["E_max_over_init"] > EXC_FACTOR
    out["E0_excursion"] = out["E0_max"] > 3.0
    return out, dict(cyc=cyc, ratio=ratio.min(axis=1), E=E, E0=E0)


def verdict(rows):
    ref = next((r for r in rows if r["sigma"] == 0), None)
    lines = []
    for r in rows:
        if r["sigma"] == 0:
            continue
        if r["E_excursion"] and r["at_contact"]:
            v = "(b) E still diverges and the driven pair has shrunk to contact: E needs a bounded normaliser even with excluded volume"
        elif r["E_excursion"] or r["E0_excursion"]:
            v = "(c) an excursion of E / E0 remains while the separations stay finite: not a normaliser effect"
        elif ref is not None and (ref["E_excursion"] or ref["collapsed"]):
            v = "(a) the excursion / collapse of the point-node run is absent with excluded volume"
        else:
            v = "no excursion with or without excluded volume at this length of run (uninformative for this network)"
        lines.append(f"  sigma={r['sigma']:g}: {v}   [E_max={r['E_max']:.3g} (point nodes: {ref['E_max']:.3g}), E0_max={r['E0_max']:.3g}, "
                     f"smallest l_t/l_t(0)={r['min_ratio']:.3g}]" if ref is not None else
                     f"  sigma={r['sigma']:g}: {v}   [E_max={r['E_max']:.3g}, E0_max={r['E0_max']:.3g}, smallest l_t/l_t(0)={r['min_ratio']:.3g}]")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_ev_2")  # this was earlier used for normal dir which is correct too
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.dir, "ev_*.jsonl")))
    if not files:
        raise SystemExit(f"no ev_*.jsonl in {a.dir}")
    rows, curves = [], {}
    for f in files:
        recs = load(f)
        if not recs:
            continue
        s, c = summarise(recs)
        s["finished"] = os.path.exists(f[:-6] + ".done")
        rows.append(s); curves[(s["label"], s["sigma"])] = c
    rows.sort(key=lambda r: (r["label"], r["sigma"]))
    cols = ["label", "sigma", "finished", "last_cycle", "min_ratio", "min_ratio_cycle", "contact_floor_of_that_target", "at_contact",
            "min_source_ratio", "E_init", "E_max", "E_max_cycle", "E0_min", "E0_max", "abs_gain_max", "n_contacts_max",
            "max_overlap_max", "e_rep_max", "min_pair_ratio_any_pair", "min_pair_dist_any_pair", "collapsed", "E_excursion", "E0_excursion"]
    with open(os.path.join(a.dir, "ev_summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for r in rows:
            w.writerow([f"{r[c]:.5g}" if isinstance(r[c], float) else r[c] for c in cols])
    json.dump(rows, open(os.path.join(a.dir, "ev_summary.json"), "w"), indent=1)

    print(f"{'net':>8} {'sigma':>5} {'done':>5} {'cyc':>5} {'min l/l0':>9} {'@cyc':>5} {'floor':>6} {'Emax':>8} {'@cyc':>5} {'E0 min-max':>12} {'|g|max':>7} {'contacts':>8} {'overlap':>8}")
    for r in rows:
        print(f"{r['label']:>8} {r['sigma']:>5g} {str(r['finished']):>5} {r['last_cycle']:>5d} {r['min_ratio']:>9.3g} {r['min_ratio_cycle']:>5d} "
              f"{r['contact_floor_of_that_target']:>6.3g} {r['E_max']:>8.3g} {r['E_max_cycle']:>5d} {r['E0_min']:>5.2f}-{r['E0_max']:<6.2f} "
              f"{r['abs_gain_max']:>7.3g} {r['n_contacts_max']:>8d} {r['max_overlap_max']:>8.3g}")
    print("\nVerdict per network (heuristic labels; read together with the table and the figure):")
    for lab in sorted({r["label"] for r in rows}):
        print(f"{lab}:")
        for l in verdict([r for r in rows if r["label"] == lab]):
            print(l)

    labs = sorted({r["label"] for r in rows})
    fig, ax = plt.subplots(2, len(labs), figsize=(3.3 * len(labs), 4.6), squeeze=False, sharex="col")
    for j, lab in enumerate(labs):
        for (l, s), c in sorted(curves.items()):
            if l != lab:
                continue
            col = SIGMA_COLOR.get(s, PURP)
            ax[0, j].plot(c["cyc"], c["ratio"], color=col, lw=1.2, label=f"$\\sigma$={s:g}")
            ax[1, j].plot(c["cyc"], c["E"], color=col, lw=1.2)
            ax[1, j].plot(c["cyc"], c["E0"], color=col, lw=1.0, ls=":")
        ax[0, j].set_yscale("log"); ax[1, j].set_yscale("log")
        ax[0, j].set_title(f"seed {lab}")
        ax[1, j].set_xlabel("aging cycle")
        ax[0, j].axhline(1, color=GREY, lw=0.6)
    ax[0, 0].set_ylabel("smallest $\\ell_t/\\ell_t(0)$ over targets")
    ax[1, 0].set_ylabel("error  (solid: $E$, dotted: $E_0$)")
    ax[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.dir, "ev_figure.png"), dpi=170); fig.savefig(os.path.join(a.dir, "ev_figure.pdf"))
    print(f"\nWrote {a.dir}/ev_summary.csv, ev_summary.json, ev_figure.png, ev_figure.pdf")


if __name__ == "__main__":
    main()
