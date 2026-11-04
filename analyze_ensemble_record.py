"""
analyze_ensemble_record.py -- what is behind the excursions of E across the whole ensemble?
    python analyze_ensemble_record.py [--dir results_ens] [--s101 path\\to\\s101.jsonl]
Writes <dir>/ens_summary.csv, ens_summary.json, ens_figure.png/.pdf, prints the counts that the manuscript needs.
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
EXC, COLL = 5.0, 0.1          # excursion: E_max > 5 E(0); collapse: a target separation < 0.1 of its initial value


def summarise(recs):
    r0 = recs[0]
    L0 = np.array(r0["target_len"])
    cyc = np.array([r["cycle"] for r in recs])
    ratio = np.array([r["target_len"] for r in recs]) / L0
    E = np.array([r["error"] for r in recs]); E0 = np.array([r["error_fixed_norm"] for r in recs])
    g = np.array([[x if x is not None else np.nan for x in r["gain"]] for r in recs])
    src = np.array([r["source_len"] for r in recs]) / r0["source_len"]
    i = np.unravel_index(np.argmin(ratio), ratio.shape)
    return dict(seed=r0["seed"], n_targets=r0["n_targets"], eps=r0["eps"], gamma=r0["gamma"], last_cycle=int(cyc[-1]),
                min_ratio=float(ratio.min()), min_ratio_cycle=int(cyc[i[0]]), shortest_initial_sep=float(L0.min()),
                min_source_ratio=float(src.min()), E_init=float(E[0]), E_max=float(E.max()), E_max_cycle=int(cyc[np.argmax(E)]),
                E_max_over_init=float(E.max() / E[0]), E0_min=float(E0.min()), E0_max=float(E0.max()),
                abs_gain_max=float(np.nanmax(np.abs(g)))), dict(cyc=cyc, E=E)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_ens")
    ap.add_argument("--s101", default=None, help="data/s101.jsonl of the paper, to check that the re-recording reproduces E")
    a = ap.parse_args()
    files = sorted(f for f in glob.glob(os.path.join(a.dir, "*.jsonl")))
    rows, curves = [], {}
    for f in files:
        recs = [json.loads(l) for l in open(f) if l.strip()]
        if not recs:
            continue
        s, c = summarise(recs)
        s["tag"] = os.path.basename(f)[:-6]; s["finished"] = os.path.exists(f[:-6] + ".done")
        rows.append(s); curves[s["tag"]] = c
    fin = [r for r in rows if r["finished"]]
    print(f"{len(rows)} files, {len(fin)} finished (only finished runs are counted below)")
    if not fin:
        raise SystemExit
    for r in fin:
        r["collapse"] = r["min_ratio"] < COLL
        r["excursion"] = r["E_max_over_init"] > EXC
        r["E0_excursion"] = r["E0_max"] > 3.0
    cols = ["tag", "seed", "n_targets", "eps", "gamma", "last_cycle", "shortest_initial_sep", "min_ratio", "min_ratio_cycle",
            "min_source_ratio", "E_init", "E_max", "E_max_cycle", "E_max_over_init", "E0_min", "E0_max", "abs_gain_max",
            "collapse", "excursion", "E0_excursion"]
    with open(os.path.join(a.dir, "ens_summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for r in sorted(fin, key=lambda r: r["min_ratio"]):
            w.writerow([f"{r[c]:.5g}" if isinstance(r[c], float) else r[c] for c in cols])
    json.dump(fin, open(os.path.join(a.dir, "ens_summary.json"), "w"), indent=1)

    n = len(fin)
    mr = np.array([r["min_ratio"] for r in fin])
    nex = sum(r["excursion"] for r in fin); nco = sum(r["collapse"] for r in fin)
    both = sum(r["excursion"] and r["collapse"] for r in fin)
    print(f"\nnetworks: {n}")
    print(f"excursion of E (E_max > {EXC:g} x E(0)): {nex}   seeds {[r['seed'] for r in fin if r['excursion']]}")
    print(f"collapse (a target separation < {COLL:g} of its initial value): {nco}   seeds {[r['seed'] for r in fin if r['collapse']]}")
    print(f"  excursion AND collapse: {both};  collapse without excursion of E: {nco - both};  excursion without collapse: {nex - both}")
    print(f"smallest ratio  l_t/l_t(0) over the ensemble: min {mr.min():.3g}, 5th pct {np.percentile(mr, 5):.3g}, median {np.median(mr):.3g}")
    print(f"  fraction of networks with a ratio below 0.5: {np.mean(mr < 0.5):.2f};  below 0.9: {np.mean(mr < 0.9):.2f}")
    print(f"E0 range over the ensemble: min {min(r['E0_min'] for r in fin):.3g}, max {max(r['E0_max'] for r in fin):.3g};  "
          f"networks with E0_max > 3: {[r['seed'] for r in fin if r['E0_excursion']]}")
    print(f"largest absolute gain: {max(r['abs_gain_max'] for r in fin):.3g}")
    print("\nNetworks with an excursion of E, and what is behind each:")
    for r in sorted((r for r in fin if r["excursion"]), key=lambda r: -r["E_max_over_init"]):
        kind = "COLLAPSE of a target pair" if r["collapse"] else ("genuine: E0 also large" if r["E0_excursion"] else "excursion of E without collapse and without large E0")
        print(f"  seed {r['seed']:>8} nT={r['n_targets']:>2}  E_max/E(0)={r['E_max_over_init']:.3g} at cycle {r['E_max_cycle']}  "
              f"min ratio={r['min_ratio']:.3g}  E0_max={r['E0_max']:.3g}  |g|max={r['abs_gain_max']:.3g}   -> {kind}")

    if a.s101 and os.path.exists(a.s101):
        old = {}
        for l in open(a.s101):
            d = json.loads(l); old[d["seed"]] = {t["cycle"]: t["error"] for t in d["trajectory"]}
        worst, nbad, ncomp = 0.0, 0, 0
        for r in fin:
            if r["seed"] not in old or r["eps"] != 0.04 or r["gamma"] != 0.012:
                continue
            new = dict(zip(curves[r["tag"]]["cyc"].tolist(), curves[r["tag"]]["E"].tolist()))
            diffs = [abs(new[c] - old[r["seed"]][c]) / old[r["seed"]][c] for c in new if c in old[r["seed"]]]
            if diffs:
                ncomp += 1; worst = max(worst, max(diffs)); nbad += max(diffs) > 0.05
        print(f"\nReproduction of data/s101.jsonl: {ncomp} networks compared; largest relative difference of E at a common cycle "
              f"{worst:.3g}; networks differing by more than 5% at some cycle: {nbad}")

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))
    div = np.array([r["excursion"] for r in fin])
    ax[0].hist(np.log10(mr), bins=24, color=GREY)
    for r in fin:
        if r["excursion"]:
            ax[0].axvline(np.log10(r["min_ratio"]), color=VERM, lw=1.2)
    ax[0].set_xlabel("log$_{10}$ of the smallest $\\ell_t/\\ell_t(0)$ over targets and cycles"); ax[0].set_ylabel("networks")
    ax[0].set_title("(a) shrinkage of target pairs (vermillion: networks with an excursion of $E$)")
    ax[0].title.set_fontsize(7)
    eo = np.array([r["E_max_over_init"] for r in fin])
    ax[1].scatter(mr[~div], eo[~div], s=9, color=GREY, label="no excursion")
    ax[1].scatter(mr[div], eo[div], s=14, color=VERM, label="excursion of $E$")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].axhline(EXC, color=GREY, lw=0.6, ls=":"); ax[1].axvline(COLL, color=GREY, lw=0.6, ls=":")
    ax[1].set_xlabel("smallest $\\ell_t/\\ell_t(0)$"); ax[1].set_ylabel("$E_{\\max}/E(0)$")
    ax[1].set_title("(b) excursion of $E$ against shrinkage"); ax[1].title.set_fontsize(7)
    ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.dir, "ens_figure.png"), dpi=170); fig.savefig(os.path.join(a.dir, "ens_figure.pdf"))
    print(f"\nWrote {a.dir}/ens_summary.csv, ens_summary.json, ens_figure.png, ens_figure.pdf")


if __name__ == "__main__":
    main()
