#!/usr/bin/env python3
"""
make_fig_lfp_inhib.py

Motivation
----------
Two questions the population-rate analyses leave open are answered by the
synaptic-current recordings: (i) does the drive-dependent reorganisation of
the ~80 Hz rhythm, including the growth of the f0/2 component, survive in a
current-based LFP proxy, i.e. in what an electrode would record, and (ii)
does a large population event recruit a larger inhibitory volley, as the
cycle-skipping account assumes. This script draws both answers in one
two-column figure from the outputs of lfp_proxy_spectra.py and
inhibitory_recruitment.py; it computes nothing new.

What it shows
-------------
  A-D  LFP-proxy power spectral density (Welch) of each excitatory population,
       the four representative drives overlaid; dotted/dashed lines at 80 and
       40 Hz. Mean over realizations when more than one is present.
  E    LFP-proxy subharmonic power ratio P(f0/2)/P(f0) versus background rate
       for the four excitatory populations (mean +/- SEM over realizations),
       the current-domain counterpart of Fig. 4A.
  F    Event-triggered average of the inhibitory synaptic current onto one
       excitatory population (--eta-pop), low- versus high-amplitude-tercile
       events, at two drives (--eta-rates); mean +/- SEM over realizations.

Usage
-----
  python make_fig_lfp_inhib.py --lfp lfp_out --inhib inhib_out --out figures
  python make_fig_lfp_inhib.py --lfp lfp_out --inhib inhib_out --proxy rws \
      --psd-rates 5.83 9.17 12.08 17.92 --eta-pop L23E --eta-rates 9.17 17.92
  python make_fig_lfp_inhib.py --selftest          # renders from synthetic inputs
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

E_POPS = ["L23E", "L4E", "L5E", "L6E"]
LAYER_COL = {"L23E": "#0072B2", "L4E": "#D55E00", "L5E": "#009E73", "L6E": "#CC79A7"}
DRIVE_COL = ["#3b4cc0", "#2a9d8f", "#e9a03b", "#c0392b"]      # low -> high drive


def nearest(avail, target, tol=0.3):
    avail = np.asarray(sorted(set(avail)), dtype=float)
    j = int(np.argmin(np.abs(avail - target)))
    if abs(avail[j] - target) > tol:
        raise SystemExit(f"no drive within {tol} of {target}; available: {avail.tolist()}")
    return float(avail[j])


def load_lfp(lfp_dir, proxy):
    psd = np.load(Path(lfp_dir) / f"lfp_{proxy}_psd.npz")
    met = pd.read_csv(Path(lfp_dir) / f"lfp_{proxy}_metrics.csv")
    spectra = {}                                   # (rate, pop) -> list of (f, P)
    for k in psd.files:
        r, tr, pop = k.split("|")
        arr = psd[k]
        spectra.setdefault((float(r), pop), []).append((arr[:, 0], arr[:, 1]))
    return spectra, met


def load_inhib(inhib_dir):
    z = np.load(Path(inhib_dir) / "inhib_eta.npz")
    lags = z["lags"]
    etas = {}                                      # (rate, pop, terc) -> list of arrays
    for k in z.files:
        if k == "lags":
            continue
        r, tr, pop, terc = k.split("|")
        etas.setdefault((float(r), pop, terc), []).append(z[k])
    summ = pd.read_csv(Path(inhib_dir) / "inhib_summary.csv")
    return lags, etas, summ


def draw(spectra, met, lags, etas, summ, psd_rates, eta_pop, eta_rates, out, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8,
                         "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.4), dpi=300)
    ax_psd = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]]
    ax_ratio, ax_eta = axes[1, 1], axes[1, 2]

    # ---- A-D: PSDs -----------------------------------------------------------
    for ax, pop in zip(ax_psd, E_POPS):
        for r, col in zip(psd_rates, DRIVE_COL):
            entries = spectra.get((r, pop))
            if not entries:
                continue
            f = entries[0][0]
            P = np.mean([e[1] for e in entries], axis=0)
            ax.semilogy(f, P, color=col, lw=0.9, label=f"{r:g}")
        ax.axvline(80, ls=":", c="0.3", lw=0.6)
        ax.axvline(40, ls="--", c="0.3", lw=0.6)
        ax.set_xlim(0, 150)
        ax.set_title(pop.replace("L23", "L2/3"), fontsize=8, pad=2)
        ax.set_xlabel("frequency [Hz]")
    ax_psd[0].set_ylabel("LFP-proxy PSD [pA$^2$/Hz]")
    ax_psd[3].set_ylabel("LFP-proxy PSD [pA$^2$/Hz]")
    ax_psd[0].legend(title="spikes/s", frameon=False, loc="upper right", handlelength=1.5, borderaxespad=0.2)

    # ---- E: subharmonic ratio vs drive ----------------------------------------
    for pop in E_POPS:
        g = met[met.population == pop].groupby("rate")["ratio"]
        m, s = g.mean(), g.sem()
        ax_ratio.errorbar(m.index, m.values, yerr=np.nan_to_num(s.values), marker="o", ms=2.5,
                          lw=0.9, capsize=1.5, elinewidth=0.6, color=LAYER_COL[pop],
                          label=pop.replace("L23", "L2/3"))
    ax_ratio.set_xlabel("background rate [spikes/s]")
    ax_ratio.set_ylabel(r"LFP proxy $P(f_0/2)\,/\,P(f_0)$")
    ax_ratio.set_ylim(bottom=0)
    ax_ratio.legend(frameon=False, ncol=2, handlelength=1.5, columnspacing=0.8)

    # ---- F: event-triggered inhibitory current ---------------------------------
    styles = [("low", "-", 0.55), ("high", "-", 1.0)]
    for r, col in zip(eta_rates, [DRIVE_COL[1], DRIVE_COL[3]]):
        for terc, ls, alpha in styles:
            rows = etas.get((r, eta_pop, terc))
            if not rows:
                continue
            base = np.mean([row[(lags >= -10) & (lags < -5)].mean() for row in rows])
            m = np.mean(rows, axis=0) - base
            s = np.std(rows, axis=0) / np.sqrt(len(rows))
            lw = 0.8 if terc == "low" else 1.3
            ax_eta.plot(lags, m, color=col, lw=lw, alpha=alpha, ls=ls,
                        label=f"{r:g} spikes/s, {terc}")
            if len(rows) > 1:
                ax_eta.fill_between(lags, m - s, m + s, color=col, alpha=0.15, lw=0)
    ax_eta.axvline(0, color="0.4", lw=0.6, ls=":")
    ax_eta.axhline(0, color="0.4", lw=0.6, ls=":")
    ax_eta.set_xlim(-5, 32)          # keeps the ~27 ms volley after a skipped cycle in view
    ax_eta.set_xlabel("time from event [ms]")
    ax_eta.set_ylabel(r"$|I_{\rm in}|$ onto " + eta_pop.replace("L23", "L2/3") + " [pA/neuron]")
    ax_eta.legend(frameon=False, handlelength=1.5, loc="upper right", title="drive, amplitude tercile", title_fontsize=6.5)

    for ax, lab in zip([*ax_psd, ax_ratio, ax_eta], "ABCDEF"):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.text(-0.2, 1.04, lab, transform=ax.transAxes, fontsize=10, fontweight="bold")
    fig.tight_layout(w_pad=1.0, h_pad=1.2)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    print("wrote", out / f"{stem}.png")

    # numbers for the caption/text
    ratio_tab = met.groupby(["population", "rate"])["ratio"].mean().unstack("rate").round(2)
    print("\nLFP-proxy subharmonic ratio (mean over realizations):\n", ratio_tab.to_string())
    for r in eta_rates:
        hi, lo = etas.get((r, eta_pop, "high")), etas.get((r, eta_pop, "low"))
        if hi and lo:
            b = lambda rows: np.mean(rows, axis=0) - np.mean([x[(lags >= -10) & (lags < -5)].mean() for x in rows])
            print(f"ETA peak ratio high/low, {eta_pop} at {r:g} spikes/s: {b(hi).max() / b(lo).max():.2f}")


def selftest(tmp):
    rng = np.random.default_rng(1)
    tmp = Path(tmp); (tmp / "lfp").mkdir(); (tmp / "inhib").mkdir()
    f = np.linspace(0, 200, 401)
    psd, rows = {}, []
    for r in (5.83, 9.17, 12.08, 17.92):
        for pop in E_POPS:
            P = 1e2 / (1 + f) + 50 * np.exp(-(f - 80) ** 2 / 8) + (r - 5) * 8 * np.exp(-(f - 40) ** 2 / 8)
            psd[f"{r:g}|0|{pop}"] = np.column_stack([f, P])
            rows.append(dict(rate=r, trial=0, population=pop, f0=80.0, P_f0=1.0,
                             P_f0_half=(r - 5) / 10, ratio=(r - 5) / 10 + rng.normal(0, 0.02)))
    np.savez(tmp / "lfp" / "lfp_rws_psd.npz", **psd)
    pd.DataFrame(rows).to_csv(tmp / "lfp" / "lfp_rws_metrics.csv", index=False)
    lags = np.arange(-10, 41, 1.0)
    etas = {"lags": lags}
    for r in (9.17, 17.92):
        for terc, a in (("low", 1.0), ("high", 1.5 if r < 10 else 2.5)):
            etas[f"{r:g}|0|L23E|{terc}"] = 100 + a * 300 * np.exp(-((lags - 2) / 1.5) ** 2) - 40 * a * np.exp(-((lags - 8) / 3) ** 2)
    np.savez(tmp / "inhib" / "inhib_eta.npz", **etas)
    pd.DataFrame([dict(rate=9.17, trial=0, population="L23E")]).to_csv(tmp / "inhib" / "inhib_summary.csv", index=False)
    spectra, met = load_lfp(tmp / "lfp", "rws")
    lags, etas_l, summ = load_inhib(tmp / "inhib")
    draw(spectra, met, lags, etas_l, summ, [5.83, 9.17, 12.08, 17.92], "L23E", [9.17, 17.92], tmp / "fig", "fig_selftest")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("Usage")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lfp", default="lfp_out")
    ap.add_argument("--inhib", default="inhib_out")
    ap.add_argument("--proxy", default="rws")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--stem", default="fig12_lfp_inhib")
    ap.add_argument("--psd-rates", nargs=4, type=float, default=[6, 9, 12, 18])
    ap.add_argument("--eta-pop", default="L23E")
    ap.add_argument("--eta-rates", nargs=2, type=float, default=[9, 18])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            sys.exit(selftest(tmp))
    spectra, met = load_lfp(a.lfp, a.proxy)
    lags, etas, summ = load_inhib(a.inhib)
    avail_psd = [k[0] for k in spectra]
    avail_eta = [k[0] for k in etas]
    psd_rates = [nearest(avail_psd, r) for r in a.psd_rates]
    eta_rates = [nearest(avail_eta, r) for r in a.eta_rates]
    draw(spectra, met, lags, etas, summ, psd_rates, a.eta_pop, eta_rates, Path(a.out), a.stem)


if __name__ == "__main__":
    main()
