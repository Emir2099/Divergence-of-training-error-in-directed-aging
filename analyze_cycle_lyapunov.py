"""
analyze_cycle_lyapunov.py -- finite-time growth rates of the cycle map against the events in the error.
    python analyze_cycle_lyapunov.py [--dir results_cyc] [--window 20]
Writes <dir>/cyc_summary.json, cyc_figure.png/.pdf and prints, per network, the finite-time exponents lambda_k (per cycle) and how
they relate to the excursion of E, to E0 and to the shrinkage of the target pairs.
"""
import argparse
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
NSUB = 8


def rolling_mean(x, w):
    """Mean over the last w cycles ending at each cycle (trailing window); NaN until w cycles are available."""
    c = np.cumsum(np.insert(x, 0, 0.0, axis=0), axis=0)
    out = np.full_like(x, np.nan, dtype=float)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_cyc")
    ap.add_argument("--window", type=int, default=20)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.dir, "cyc_*.jsonl")))
    if not files:
        raise SystemExit(f"no cyc_*.jsonl in {a.dir}")
    nets, summary = [], []
    for f in files:
        recs = [json.loads(l) for l in open(f) if l.strip()]
        steps = [r for r in recs if r["kind"] == "step"]; meas = [r for r in recs if r["kind"] == "meas"]
        if len(steps) < a.window + 1 or not meas:
            print(f"{os.path.basename(f)}: too short ({len(steps)} steps), skipped"); continue
        tag = os.path.basename(f)[4:-6]
        c_s = np.array([r["cycle"] for r in steps]); logR = np.array([r["logR"] for r in steps])      # (n, K)
        lam_w = rolling_mean(logR, a.window)
        c_m = np.array([r["cycle"] for r in meas])
        E = np.array([r["error"] for r in meas]); E0 = np.array([r["error_fixed_norm"] for r in meas])
        L0 = np.array(meas[0]["target_len"]); ratio = (np.array([r["target_len"] for r in meas]) / L0).min(axis=1)
        lamS = np.array([r["lam_max_S"] for r in meas]); gamma = meas[0]["gamma"]
        naive = NSUB * gamma * (lamS - 1.0)
        i_e = int(np.argmax(E)); c_e = int(c_m[i_e])
        pre = (c_s > c_e - 100) & (c_s <= c_e)
        l1 = logR[:, 0]
        s = dict(net=tag, K=logR.shape[1], n_cycles=int(len(c_s) ), gamma=gamma, window=a.window,
                 lambda_mean_all=[float(x) for x in logR.mean(axis=0)],
                 lambda1_window_max=float(np.nanmax(lam_w[:, 0])), lambda1_window_max_cycle=int(c_s[np.nanargmax(lam_w[:, 0])]),
                 lambda1_window_min=float(np.nanmin(lam_w[:, 0])),
                 frac_windows_lambda1_positive=float(np.mean(lam_w[a.window - 1:, 0] > 0)),
                 E_max=float(E.max()), E_max_over_init=float(E.max() / E[0]), E_max_cycle=c_e,
                 E0_max=float(E0.max()), min_ratio=float(ratio.min()), min_ratio_cycle=int(c_m[np.argmin(ratio)]),
                 lambda1_mean_100_cycles_before_Emax=float(l1[pre].mean()) if pre.any() else None,
                 lam_max_S_max=float(lamS.max()), naive_growth_max=float(naive.max()))
        summary.append(s)
        nets.append(dict(tag=tag, c_s=c_s, lam_w=lam_w, c_m=c_m, E=E, E0=E0, ratio=ratio, naive=naive))
    json.dump(summary, open(os.path.join(a.dir, "cyc_summary.json"), "w"), indent=1)

    print(f"finite-time exponents lambda_k = mean over the cycles of log(growth of direction k in one cycle); window = {a.window} cycles")
    print("lambda > 0: perturbations of the rest lengths grow under one aging cycle.  The exponents are per CYCLE.\n")
    for s in summary:
        print(f"== {s['net']}  (K={s['K']}, {s['n_cycles']} cycles, gamma={s['gamma']})")
        print("   mean lambda_k over the whole run:      " + "  ".join(f"{x:+.2e}" for x in s["lambda_mean_all"]))
        print(f"   lambda_1 over {a.window}-cycle windows:   max {s['lambda1_window_max']:+.2e} (window ending at cycle {s['lambda1_window_max_cycle']}), "
              f"min {s['lambda1_window_min']:+.2e}, positive in {100 * s['frac_windows_lambda1_positive']:.0f}% of windows")
        print(f"   error: E_max/E(0) = {s['E_max_over_init']:.3g} at cycle {s['E_max_cycle']};  E0_max = {s['E0_max']:.3g};  "
              f"smallest l_t/l_t(0) = {s['min_ratio']:.3g} at cycle {s['min_ratio_cycle']}")
        if s["lambda1_mean_100_cycles_before_Emax"] is not None:
            print(f"   mean lambda_1 in the 100 cycles up to the maximum of E: {s['lambda1_mean_100_cycles_before_Emax']:+.2e}")
        print(f"   undriven criterion: max lambda_max(S) = {s['lam_max_S_max']:.3g}; naive per-cycle growth {NSUB}*gamma*(lambda_max(S)-1) reaches "
              f"{s['naive_growth_max']:+.2e} (reference only: it assumes all {NSUB} substeps have the undriven Jacobian)")
    print("\nHow to read it: if lambda_1 is positive before and during an excursion of E (or of E0) and about zero or negative elsewhere,")
    print("the cycle map has a growing direction that the undriven criterion misses; if lambda_1 is about the same in quiet and in")
    print("excursion windows, the excursion is not a linear instability of the map (consistent with a collapse of the normaliser).")

    fig, ax = plt.subplots(3, len(nets), figsize=(3.3 * len(nets), 6.2), squeeze=False, sharex="col")
    cols = [VERM, BLUE, GREEN, ORANGE, PURP]
    for j, n in enumerate(nets):
        for k in range(n["lam_w"].shape[1]):
            ax[0, j].plot(n["c_s"], n["lam_w"][:, k], color=cols[k % 5], lw=1.1, label=f"$\\lambda_{k + 1}$")
        ax[0, j].plot(n["c_m"], n["naive"], color=GREY, lw=0.9, ls=":", label=f"naive {NSUB}$\\gamma(\\lambda_{{\\max}}(S)-1)$")
        ax[0, j].axhline(0, color="k", lw=0.5)
        ax[0, j].set_title(n["tag"])
        ax[1, j].plot(n["c_m"], n["E"], color=VERM, lw=1.1, label="$E$"); ax[1, j].plot(n["c_m"], n["E0"], color=BLUE, lw=1.1, ls=":", label="$E_0$")
        ax[1, j].set_yscale("log")
        ax[2, j].plot(n["c_m"], n["ratio"], color="k", lw=1.1); ax[2, j].set_yscale("log"); ax[2, j].set_xlabel("aging cycle")
    ax[0, 0].set_ylabel(f"finite-time growth rate\nper cycle ({a.window}-cycle window)"); ax[1, 0].set_ylabel("error"); ax[2, 0].set_ylabel("smallest $\\ell_t/\\ell_t(0)$")
    ax[0, 0].legend(frameon=False); ax[1, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.dir, "cyc_figure.png"), dpi=170); fig.savefig(os.path.join(a.dir, "cyc_figure.pdf"))
    print(f"\nWrote {a.dir}/cyc_summary.json, cyc_figure.png, cyc_figure.pdf")


if __name__ == "__main__":
    main()
