"""
analyze_onset_map.py -- is n* = D / (gamma eps^2) in the second network, and what is behind each divergence?
    python analyze_onset_map.py [--dir results_onset]
Writes <dir>/onset_table.csv, onset_summary.json, onset_figure.png/.pdf and prints the table, D per run, the free fit of
ln n* on ln eps and ln gamma with standard errors, and the runs for which the regularity predicts an onset inside the budget
but none was found.
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
FACTOR = 5.0


def analyse(recs):
    r0 = recs[0]
    cyc = np.array([r["cycle"] for r in recs]); E = np.array([r["error"] for r in recs]); E0 = np.array([r["error_fixed_norm"] for r in recs])
    L0 = np.array(r0["target_len"]); ratio = (np.array([r["target_len"] for r in recs]) / L0).min(axis=1)
    g = np.array([[x if x is not None else np.nan for x in r["gain"]] for r in recs])
    idx = np.where(E > FACTOR * E[0])[0]
    n_star = int(cyc[idx[0]]) if len(idx) else None
    out = dict(seed=r0["seed"], eps=r0["eps"], gamma=r0["gamma"], n_targets=r0["n_targets"], last_cycle=int(cyc[-1]),
               E_init=float(E[0]), E_max=float(E.max()), diverged=bool(len(idx)), n_star=n_star,
               D_run=(n_star * r0["gamma"] * r0["eps"] ** 2) if n_star else None,
               min_ratio=float(ratio.min()), E0_max=float(E0.max()), abs_gain_max=float(np.nanmax(np.abs(g))))
    if n_star:
        out["min_ratio_at_nstar"] = float(ratio[idx[0]])
    out["mechanism"] = None
    if out["diverged"]:
        out["mechanism"] = ("collapse" if out["min_ratio"] < 0.2 and out["E0_max"] < 3 else
                            "genuine" if out["E0_max"] >= 3 and out["min_ratio"] >= 0.2 else "mixed/other")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_onset")
    a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.dir, "onset_*.jsonl"))):
        recs = [json.loads(l) for l in open(f) if l.strip()]
        if len(recs) < 2:
            continue
        r = analyse(recs); r["finished"] = os.path.exists(f[:-6] + ".done"); rows.append(r)
    if not rows:
        raise SystemExit(f"no onset_*.jsonl in {a.dir}")
    rows.sort(key=lambda r: (r["eps"], r["gamma"]))
    cols = ["seed", "eps", "gamma", "finished", "last_cycle", "diverged", "n_star", "D_run", "mechanism", "min_ratio", "E0_max", "abs_gain_max", "E_max"]
    with open(os.path.join(a.dir, "onset_table.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for r in rows:
            w.writerow([f"{r[c]:.5g}" if isinstance(r[c], float) else r[c] for c in cols])
    print(f"{'eps':>6} {'gamma':>6} {'done':>5} {'last':>5} {'diverged':>8} {'n*':>6} {'n* g e^2':>9} {'mechanism':>10} {'min l/l0':>9} {'E0max':>6} {'|g|max':>7}")
    for r in rows:
        ns = str(r["n_star"]) if r["n_star"] else "-"
        dr = "%.4f" % r["D_run"] if r["D_run"] else "-"
        print(f"{r['eps']:>6g} {r['gamma']:>6g} {str(r['finished']):>5} {r['last_cycle']:>5d} {str(r['diverged']):>8} "
              f"{ns:>6} {dr:>9} {str(r['mechanism'] or '-'):>10} {r['min_ratio']:>9.3g} {r['E0_max']:>6.2f} {r['abs_gain_max']:>7.3g}")
    div = [r for r in rows if r["diverged"]]
    summ = dict(n_runs=len(rows), n_diverged=len(div), seed=rows[0]["seed"])
    print(f"\nseed {rows[0]['seed']}: {len(rows)} runs, {len(div)} diverged (E > {FACTOR:g} x E(0)); mechanisms: "
          f"{ {m: sum(r['mechanism'] == m for r in div) for m in sorted({r['mechanism'] for r in div})} }")
    if div:
        D = np.array([r["D_run"] for r in div])
        summ.update(D_median=float(np.median(D)), D_min=float(D.min()), D_max=float(D.max()), D_spread=float(D.max() / D.min()))
        print(f"n* gamma eps^2 over the {len(div)} diverging runs: median {np.median(D):.4f}, range {D.min():.4f}-{D.max():.4f} (factor {D.max() / D.min():.2f}).")
        print("   (seed 1000005: 14 of 15 runs between 0.0084 and 0.0116, median 0.0096, a factor 1.4; the measurement interval limits small n*)")
    if len(div) >= 5 and len({r["eps"] for r in div}) >= 2 and len({r["gamma"] for r in div}) >= 2:
        X = np.array([[1.0, np.log(r["eps"]), np.log(r["gamma"])] for r in div]); y = np.log([r["n_star"] for r in div])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta; s2 = float(res @ res) / (len(y) - 3)
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
        print(f"free fit  ln n* = a + b ln eps + c ln gamma  ({len(div)} runs):  b = {beta[1]:+.2f} +- {se[1]:.2f},  c = {beta[2]:+.2f} +- {se[2]:.2f}"
              f"   (the regularity has b = -2, c = -1; seed 1000005 gave -1.67 +- 0.31 and -0.80 +- 0.12)")
        summ.update(fit_b=float(beta[1]), fit_b_se=float(se[1]), fit_c=float(beta[2]), fit_c_se=float(se[2]))
    else:
        print("free fit not done (needs >= 5 diverging runs with at least two values of eps and of gamma)")
    if div:
        Dm = float(np.median([r["D_run"] for r in div]))
        miss = [r for r in rows if not r["diverged"] and r["finished"] and Dm / (r["gamma"] * r["eps"] ** 2) < r["last_cycle"]]
        print(f"\nruns that did not diverge although the regularity with D = {Dm:.4f} predicts n* inside the budget: {len(miss)}")
        for r in miss:
            print(f"   eps={r['eps']:g} gamma={r['gamma']:g}: predicted n* = {Dm / (r['gamma'] * r['eps'] ** 2):.0f}, budget {r['last_cycle']}")
        summ["n_predicted_but_absent"] = len(miss)
    json.dump(dict(summary=summ, runs=rows), open(os.path.join(a.dir, "onset_summary.json"), "w"), indent=1)

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.1))
    for r in rows:
        if r["diverged"]:
            col = VERM if r["mechanism"] == "collapse" else BLUE if r["mechanism"] == "genuine" else ORANGE
            ax[0].plot(r["eps"], r["gamma"], "x", color=col, ms=7, mew=1.6)
        else:
            ax[0].plot(r["eps"], r["gamma"], "o", mfc="none", color=GREY, ms=6)
    ax[0].set_xlabel("$\\epsilon$"); ax[0].set_ylabel("$\\gamma$"); ax[0].set_title("(a) x: diverged (vermillion: collapse, blue: genuine); o: no")
    ax[0].title.set_fontsize(7)
    if div:
        x = np.array([1 / (r["gamma"] * r["eps"] ** 2) for r in div]); yv = np.array([r["n_star"] for r in div])
        for r, xi, yi in zip(div, x, yv):
            ax[1].plot(xi, yi, "o", color=VERM if r["mechanism"] == "collapse" else BLUE if r["mechanism"] == "genuine" else ORANGE, ms=5)
        xs = np.geomspace(x.min() * 0.8, x.max() * 1.25, 20)
        ax[1].plot(xs, summ["D_median"] * xs, color="k", lw=0.9, label=f"$n^*=D/(\\gamma\\epsilon^2)$, $D$={summ['D_median']:.4f} (median)")
        ax[1].plot(xs, 0.0096 * xs, color=GREY, lw=0.9, ls="--", label="$D$=0.0096 (seed 1000005)")
        ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].legend(frameon=False)
    ax[1].set_xlabel("$1/(\\gamma\\epsilon^2)$"); ax[1].set_ylabel("$n^*$"); ax[1].set_title("(b) onset cycle"); ax[1].title.set_fontsize(7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.dir, "onset_figure.png"), dpi=170); fig.savefig(os.path.join(a.dir, "onset_figure.pdf"))
    print(f"\nWrote {a.dir}/onset_table.csv, onset_summary.json, onset_figure.png, onset_figure.pdf")


if __name__ == "__main__":
    main()
