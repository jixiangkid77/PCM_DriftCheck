"""Build the figures of the PCM_DriftCheck brief (TCAS-II).

Location (frozen by E1_Design_Spec section 1):
    D:\\Project\\PCM_DriftCheck\\figures\\make_paper_figs.py
Figures are written next to this script unless --out says otherwise, and are
named fig1_construction.pdf through fig7_transition.pdf (plus .png), by figure number in the brief.

Usage (KID, formal records):
    python figures\\make_paper_figs.py ^
        --e1 results\\E1\\20260829_1239 --e2 results\\E2\\20260829_1344 ^
        --e3 results\\E3\\20260830_0124 --e4 results\\E4\\20260830_0150 ^
        --e5 results\\E5\\20260830_0349 --e6 results\\E6\\20260830_1309

One figure only (fig1_construction needs no record; it is a schematic):
    python figures\\make_paper_figs.py --only fig1_construction

Outputs, named by figure number as they appear in the brief:
    fig1_construction  construction schematic and checker datapath   (data free)
    fig2_residual_law  fault-free residual law                       (E1)
    fig3_checker  digitized checker, band error and bit true    (E6)
    fig4_validation  variance-law validation and budget backtest   (E2 + E3)
    fig5_n90  detection latency over (t, sigma_nu)          (E3)
    fig6_motivation  the three decision rules                      (E4)
    fig7_transition  sign-flip confusability edge                  (E5)
Record directories are required only for the figures actually selected.
"""
from __future__ import annotations

import argparse
import os
import sys

import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

RC = {
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "lines.linewidth": 1.1, "figure.dpi": 200, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    # Minor ticks are switched off everywhere (KID, 2026-09-01); on log axes
    # matplotlib draws them from the locator, so _save clears them per axes.
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.pad": 1.8, "ytick.major.pad": 1.8,
}
MAIN = dict(encoding="b", sigma_nu=0.02, m=16)

# Shared schematic colours (Fig. 1).
CK_FILL, CK_EDGE = "#9ecae1", "#2171b5"   # checksum column
CMP_FILL = "#fee0d2"                       # comparator stage
PWL_FILL = "#deebf7"                       # band register


def _save(fig, out, name):
    for a in fig.axes:                # single point of control: no minor ticks
        a.minorticks_off()
    fig.savefig(os.path.join(out, name + ".pdf"))
    fig.savefig(os.path.join(out, name + ".png"))
    plt.close(fig)
    print("wrote", name)


def fig1_construction(out):
    """Fig. 1: the column-group checksum construction and the checker datapath.

    Data free. Panel (a) illustrates Section II-D: one checksum column per
    group of m data columns, storing u_i = clip(lambda * sum_j w_ij).
    Panel (b) is the post-ADC datapath of Section III-E: adder tree, right
    shift realizing lambda = 1/m, subtraction, absolute value and comparison
    against the band, alarm register; the piecewise-linear evaluator of
    f(ln t) is the only approximation in the checker.

    Geometry is computed from named constants so that nothing is placed by
    trial and error; every element is kept inside the axes box.
    """
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(3.45, 1.42), width_ratios=[1.06, 1.0])
    for a in (ax, bx):
        a.set_xlim(0, 100)
        a.set_ylim(0, 100)
        a.axis("off")
        a.grid(False)
        a.set_aspect("auto")

    # ---------------- (a) crossbar with one checksum column ---------------
    N_ROW, N_DATA = 4, 4                 # illustrative; the array is 128x128
    n_col = N_DATA + 1
    CELL_W, CELL_H = 8.0, 8.5            # cell footprint in axes units
    X_FIRST, Y_TOP = 22.0, 68.0          # centre of the top-left cell
    DX = (92.0 - X_FIRST) / (n_col - 1)  # column pitch, last col ends at 92
    DY = 15.0                            # row pitch

    col_x = [X_FIRST + j * DX for j in range(n_col)]
    row_y = [Y_TOP - i * DY for i in range(N_ROW)]

    for i, y in enumerate(row_y):        # wordlines and their input labels
        ax.plot([X_FIRST - CELL_W, col_x[-1] + CELL_W], [y, y],
                color="0.55", lw=0.6, zorder=1)
        ax.text(X_FIRST - CELL_W - 3.0, y, rf"$x_{{{i + 1}}}$",
                ha="right", va="center", fontsize=6.5)
    for x in col_x:                      # bitlines
        ax.plot([x, x], [row_y[0] + CELL_H, row_y[-1] - CELL_H],
                color="0.55", lw=0.6, zorder=1)
    for i, y in enumerate(row_y):        # cells, checksum column shaded
        for j, x in enumerate(col_x):
            is_ck = j == n_col - 1
            ax.add_patch(Rectangle(
                (x - CELL_W / 2, y - CELL_H / 2), CELL_W, CELL_H,
                facecolor=CK_FILL if is_ck else "0.92",
                edgecolor="0.25", lw=0.55, zorder=2))

    brace_y = row_y[0] + CELL_H + 6.0    # span marker over the data columns
    ax.plot([col_x[0] - CELL_W / 2, col_x[N_DATA - 1] + CELL_W / 2],
            [brace_y, brace_y], color="0.2", lw=0.8, zorder=3)
    for xe in (col_x[0] - CELL_W / 2, col_x[N_DATA - 1] + CELL_W / 2):
        ax.plot([xe, xe], [brace_y - 2.5, brace_y], color="0.2", lw=0.8)
    ax.text(0.5 * (col_x[0] + col_x[N_DATA - 1]), brace_y + 3.0,
            r"$m$ data columns", ha="center", va="bottom", fontsize=6.5)
    ax.text(col_x[-1], brace_y + 3.0, r"checksum $u_i$", ha="center",
            va="bottom", fontsize=6.5, color=CK_EDGE)

    lab_y = row_y[-1] - CELL_H - 4.0     # bitline output labels
    for j in range(N_DATA):
        ax.text(col_x[j], lab_y, rf"$y_{{{j + 1}}}$", ha="center",
                va="top", fontsize=6.5)
    ax.text(col_x[-1], lab_y, r"$y_{\mathrm{ck}}$", ha="center", va="top",
            fontsize=6.5, color=CK_EDGE)
    ax.set_title("(a) construction", fontsize=7.5, pad=2.0)

    # ---------------- (b) post-ADC checker datapath -----------------------
    # Braced sums keep the limits inline so the glyphs stay inside the boxes.
    steps = [r"adder tree ${\sum}_{j\in S}\, y_j$",
             r"shift $\gg \log_2 m$",
             r"$r = y_{\mathrm{ck}} - \lambda\,{\sum}_{j\in S} y_j$",
             r"$|r| > f(t)\;$?"]
    BOX_W, BOX_H = 62.0, 13.0
    BOX_X = 30.0                          # centre of the chain
    TOP, GAP = 84.0, 8.0                  # first box centre, gap between boxes
    centres = [TOP - k * (BOX_H + GAP) for k in range(len(steps))]

    for k, (label, yc) in enumerate(zip(steps, centres)):
        last = k == len(steps) - 1
        bx.add_patch(Rectangle(
            (BOX_X - BOX_W / 2, yc - BOX_H / 2), BOX_W, BOX_H,
            facecolor=CMP_FILL if last else "0.94",
            edgecolor="0.2", lw=0.65, zorder=2))
        bx.text(BOX_X, yc, label, ha="center", va="center", fontsize=6.0,
                zorder=3)
        if not last:                      # arrow strictly inside the gap
            y_from = yc - BOX_H / 2
            bx.annotate("", xy=(BOX_X, y_from - GAP + 0.6),
                        xytext=(BOX_X, y_from - 0.6),
                        arrowprops=dict(arrowstyle="-|>", lw=0.7,
                                        color="0.2", shrinkA=0, shrinkB=0,
                                        mutation_scale=5))

    y_cmp = centres[-1]                   # band register feeds the comparator
    pwl_x0 = BOX_X + BOX_W / 2 + 9.0
    pwl_w = 100.0 - pwl_x0 - 1.0
    bx.add_patch(Rectangle((pwl_x0, y_cmp - BOX_H / 2), pwl_w, BOX_H,
                           facecolor=PWL_FILL, edgecolor="0.2", lw=0.65,
                           zorder=2))
    bx.text(pwl_x0 + pwl_w / 2, y_cmp, "PWL\n$f(\\ln t)$", ha="center",
            va="center", fontsize=5.4, linespacing=1.1, zorder=3)
    bx.annotate("", xy=(BOX_X + BOX_W / 2 + 0.6, y_cmp),
                xytext=(pwl_x0 - 0.6, y_cmp),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.2",
                                shrinkA=0, shrinkB=0, mutation_scale=5))

    y_alarm_from = y_cmp - BOX_H / 2      # comparator to the alarm register
    bx.annotate("", xy=(BOX_X, y_alarm_from - 8.5),
                xytext=(BOX_X, y_alarm_from - 0.6),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.2",
                                shrinkA=0, shrinkB=0, mutation_scale=5))
    bx.text(BOX_X, y_alarm_from - 10.5, "alarm register", ha="center",
            va="top", fontsize=6.0)
    bx.set_title("(b) checker datapath", fontsize=7.5, pad=2.0)

    fig.tight_layout(pad=0.12, w_pad=0.6)
    _save(fig, out, "fig1_construction")


def fig4_validation(e2, e3, out):
    """Fig. 4: the variance law validated, then the deployed band backtested.

    (a) Closed-form residual scale against simulation on both readout faces
    (E2 record, encoding b, m = 16, sigma_nu = 0.02). The pooled model runs
    above the ratio-face data at late times; the conditioned model, which
    uses the measured reference scale, is drawn as well so that the gap the
    text reports is visible rather than only asserted. Lines are the model,
    markers the simulation, so no legend is needed: the two faces are
    labelled in place.
    (b) Deployed-band false-positive rate over the E3 backtest grid. The two
    faces are NOT equally sampled: 360 cells on the ratio face against 9 on
    the direct face, so the legend carries the counts and the direct points
    are drawn on top at a larger size.
    """
    df = pd.read_csv(os.path.join(e2, "model_vs_data.csv"))
    df = df[(df.encoding == MAIN["encoding"]) & (df.m == MAIN["m"]) &
            (np.isclose(df.sigma_nu, MAIN["sigma_nu"]))]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(3.45, 1.20))

    for rn, mk, col in [(0, "o", "C0"), (1, "s", "C3")]:
        d = df[df.ratio_norm == rn].sort_values("t_rel")
        ax.plot(d.t_rel, d.sigma_model, "-", lw=0.9, color=col)
        ax.plot(d.t_rel, d.sigma_data, mk, ms=2.4, mfc="none", mew=0.8,
                color=col, ls="none")
        if rn == 1:                    # conditioned on the measured scale
            ax.plot(d.t_rel, d.sigma_model_cond, ":", lw=0.9, color=col)
    ax.set_xscale("log")
    ax.set_ylim(0, 0.72)
    ax.set_xticks([1e0, 1e2, 1e4, 1e6, 1e8])
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_xlabel(r"$t/t_0$")
    ax.set_ylabel(r"$\sigma_r$")
    ax.set_title("(a) residual scale", fontsize=7)
    ax.text(40, 0.56, "ratio", fontsize=5.4, color="C3", ha="center")
    ax.text(2e6, 0.115, "direct", fontsize=5.4, color="C0", ha="center")

    bt = pd.read_csv(os.path.join(e3, "fpr_backtest.csv"))
    n_ratio = int((bt.ratio_norm == 1).sum())
    n_direct = int((bt.ratio_norm == 0).sum())
    assert n_ratio + n_direct == 369, (n_ratio, n_direct)
    d1 = bt[bt.ratio_norm == 1]
    d0 = bt[bt.ratio_norm == 0]
    bx.scatter(d1.t_rel, d1.fpr, s=2.2, marker="o", alpha=0.30, zorder=2,
               linewidths=0.0, color="C0", label="ratio, %d cells" % n_ratio)
    bx.scatter(d0.t_rel, d0.fpr, s=7.0, marker="x", zorder=4, linewidths=0.7,
               color="#d95f02", label="direct, %d cells" % n_direct)
    bx.axhline(0.01, color="k", lw=0.8, ls="--", zorder=3)
    bx.set_xscale("log")
    bx.set_yscale("log")
    bx.set_ylim(7e-4, 2.2e-1)
    bx.set_xticks([1e0, 1e2, 1e4, 1e6, 1e8])
    bx.set_yticks([1e-3, 1e-2, 1e-1])
    bx.set_xlabel(r"$t/t_0$")
    bx.set_ylabel("band FPR")
    bx.set_title("(b) budget backtest", fontsize=7)
    bx.legend(fontsize=5.0, frameon=False, loc="upper left",
              handletextpad=0.1, borderpad=0.0, labelspacing=0.1,
              scatterpoints=1)
    fig.tight_layout(pad=0.12, w_pad=0.5)
    _save(fig, out, "fig4_validation")


def fig5_n90(e3, out):
    """Fig. 5: model detection latency over the (t, sigma_nu) plane.

    The band drawn here is the one the E3 record deployed:
    f(t) = theta0_hat * rho(t) with theta0_hat the EMPIRICAL (1 - FPR0)
    quantile calibrated at t0, read from fpr_backtest.csv and interpolated
    in log sigma_nu between its four calibrated values. An earlier version
    of this figure substituted the Gaussian surrogate z0 * sigma_r for
    theta0_hat, which disagreed with the recorded p1_model by up to
    5.2e-2; the assertion below pins the surface to the record.
    Markers are the simulation grid points of the same slice.
    """
    import json
    from pcm_drift import model
    df = pd.read_csv(os.path.join(e3, "p1_curves.csv"))
    df = df[(df.encoding == "b") & (df.m == 8) & (df.c == 1) &
            (df.ratio_norm == 1)].copy()
    df["delta"] = pd.to_numeric(df.delta, errors="coerce")
    df = df.dropna(subset=["delta"])
    dmax = float(df.delta.max())
    df = df[np.isclose(df.delta, dmax)]

    bt = pd.read_csv(os.path.join(e3, "fpr_backtest.csv"))
    bt = bt[(bt.encoding == "b") & (bt.m == 8) & (bt.c == 1) &
            (bt.ratio_norm == 1)]
    cal = bt.groupby("sigma_nu").theta0.agg(["nunique", "first"])
    assert (cal["nunique"] == 1).all(), "theta0 must be one value per config"
    s_cal = cal.index.values.astype(float)
    th_cal = cal["first"].values.astype(float)

    cfg = json.load(open(os.path.join(e3, "config.json")))
    p = model.ModelParams.from_run_config({"config": cfg["physics"]})

    def p1_of(s, L):
        """Deployed-band detection probability at dispersion s, time ln t."""
        th0 = float(np.interp(np.log(s), np.log(s_cal), th_cal))
        _, sd0 = model.r_moments("b", 8, 0.0, s, p, True, lam_scale=1)
        _, sd = model.r_moments("b", 8, L, s, p, True, lam_scale=1)
        dv, dw = model.fault_dm_nodes("b", "soft_abs", dmax, L, s, p)
        return model.predict_p1("b", 8, L, s, p, True, th0 * (sd / sd0),
                                dv, dw, lam_scale=1)

    # the surface must reproduce the record wherever the record has a point
    chk = max(abs(p1_of(float(r.sigma_nu), float(np.log(r.t_rel))) - r.p1_model)
              for _, r in df.iterrows())
    assert chk < 1e-9, "surface departs from the recorded p1_model: %g" % chk

    def n90_of(p1):
        return np.log(0.1) / np.log(1 - min(max(p1, 1e-9), 0.999))

    sig = np.logspace(np.log10(s_cal.min()), np.log10(s_cal.max()), 13)
    tt = np.logspace(0, 8, 17)
    n90 = np.array([[n90_of(p1_of(float(s), float(np.log(t)))) for t in tt]
                    for s in sig])

    fig, ax = plt.subplots(figsize=(3.3, 1.36))
    lev = [10, 20, 50, 100, 300, 1000, 3000]
    cs = ax.contourf(tt, sig, n90, levels=lev, cmap="Blues",
                     norm=matplotlib.colors.LogNorm(vmin=10, vmax=3000),
                     extend="both")
    ax.contour(tt, sig, n90, levels=[20], colors="k", linewidths=1.0)
    ax.contour(tt, sig, n90, levels=[100], colors="0.30", linewidths=0.8,
               linestyles="--")
    ok = np.array([n90_of(v) <= 20 for v in df.p1_sim])
    ax.plot(df.t_rel[ok], df.sigma_nu[ok], "o", color="k", ms=3.2,
            mfc="none", mew=0.8, clip_on=False)
    ax.plot(df.t_rel[~ok], df.sigma_nu[~ok], "x", color="k", ms=2.8, mew=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.75, 1.4e8)
    ax.set_ylim(0.0042, 0.047)      # keep the extreme rows off the frame
    ax.set_xticks([1e0, 1e2, 1e4, 1e6, 1e8])
    ax.set_yticks(s_cal)
    ax.set_yticklabels(["%g" % v for v in s_cal])
    ax.set_xlabel(r"$t/t_0$")
    ax.set_ylabel(r"$\sigma_\nu$")
    # Contour meanings and marker meanings go in one legend above the panel:
    # inline contour labels were illegible at this width.
    proxies = [
        matplotlib.lines.Line2D([], [], color="k", lw=1.0),
        matplotlib.lines.Line2D([], [], color="0.30", lw=0.8, ls="--"),
        matplotlib.lines.Line2D([], [], color="k", marker="o", ls="none",
                                ms=3.2, mfc="none", mew=0.8),
        matplotlib.lines.Line2D([], [], color="k", marker="x", ls="none",
                                ms=2.8, mew=0.8)]
    ax.legend(proxies, [r"$N^{*}{=}20$", r"$N_{90}{=}100$",
                        r"sim., within $N^{*}$", r"sim., beyond $N^{*}$"],
              fontsize=5.0, frameon=False, ncols=4, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), columnspacing=0.8,
              handlelength=1.5, handletextpad=0.3, borderpad=0.0)
    cb = fig.colorbar(cs, ax=ax, pad=0.02)
    cb.set_label(r"$N_{90}$ at $\delta = %.2f$" % dmax, fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    _save(fig, out, "fig5_n90")


def fig6_motivation(e4, out):
    """Fig. 6: the three decision rules against the false-positive budget.

    Reads the E4 record. Slice: encoding b, m = 16, c = 1, sigma_nu = 0.02,
    ratio readout. The budget FPR0 = 1 percent is drawn as a line, not as a
    shaded region: the band runs slightly above it (max 0.0133 in this
    slice), and the figure must not suggest otherwise.
    """
    df = pd.read_csv(os.path.join(e4, "fpr_compare.csv"))
    df = df[(df.encoding == "b") & (df.m == 16) & (df.c == 1) &
            (np.isclose(df.sigma_nu, 0.02)) & (df.ratio_norm == 1)]
    style = {"A6": ("#e34a33", ":"), "A8": ("#b30000", "--"),
             "A10": ("#7f0000", "-."), "B_const": ("#ff8c00", (0, (5, 1))),
             "C_band": ("C0", "-")}
    lab = {"A6": "exact 6 b", "A8": "exact 8 b", "A10": "exact 10 b",
           "B_const": "constant tol.", "C_band": "band"}
    fig, ax = plt.subplots(figsize=(3.3, 1.12))
    for det in ["A6", "A8", "A10", "B_const", "C_band"]:
        d = df[df.detector == det].sort_values("t_rel")
        c, ls = style[det]
        ax.plot(d.t_rel, d.fpr, ls=ls, color=c,
                lw=1.5 if det == "C_band" else 1.0, label=lab[det])
    ax.axhline(0.01, color="k", lw=0.8, ls="--")
    ax.text(1.2, 0.0060, r"budget $\mathrm{FPR}_0 = 1\%$", fontsize=5.2,
            va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(4.2e-3, 1.5)
    ax.set_xticks([1e0, 1e2, 1e4, 1e6, 1e8])
    ax.set_xlabel(r"$t/t_0$")
    ax.set_ylabel("false-positive rate")
    # Legend outside the axes: every interior region carries a curve.
    ax.legend(fontsize=5.2, frameon=False, loc="lower center", ncols=5,
              columnspacing=0.8, handlelength=1.6, handletextpad=0.3,
              borderpad=0.0, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.12)
    _save(fig, out, "fig6_motivation")


def fig_val_n90(e2, e3, out):
    """Combined half-width panels: (a) validation, (b) N90 contours."""
    import json
    from pcm_drift import model
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(3.4, 1.5))
    small = {"labelsize": 6.5, "ticksize": 6, "legend": 5.2}
    df = pd.read_csv(os.path.join(e2, "model_vs_data.csv"))
    df = df[(df.encoding == "b") & (df.m == 16) &
            (np.isclose(df.sigma_nu, 0.02))]
    for rn, ls, mk, lab in [(0, "-", "o", "direct"), (1, "--", "s", "ratio")]:
        d = df[df.ratio_norm == rn].sort_values("t_rel")
        c = "C0" if rn == 0 else "C3"
        ax.plot(d.t_rel, d.sigma_model, ls, color=c, lw=1.0,
                label=f"model, {lab}")
        ax.plot(d.t_rel, d.sigma_data, mk, ms=2.6, mfc="none", color=c,
                ls="none", label=f"sim, {lab}")
    ax.set_xscale("log")
    ax.set_xlabel(r"$t/t_0$", fontsize=small["labelsize"])
    ax.set_ylabel(r"$\sigma_r$", fontsize=small["labelsize"])
    ax.tick_params(labelsize=small["ticksize"])
    ax.legend(fontsize=small["legend"], handletextpad=0.3, borderpad=0.25,
              labelspacing=0.2, handlelength=1.3)
    ax.set_title("(a)", fontsize=7)
    dfp = pd.read_csv(os.path.join(e3, "p1_curves.csv"))
    dfp = dfp[(dfp.encoding == "b") & (dfp.m == 8) & (dfp.c == 1) &
              (dfp.ratio_norm == 1)].copy()
    dfp["delta"] = pd.to_numeric(dfp.delta, errors="coerce")
    dfp = dfp.dropna(subset=["delta"])
    dmax = float(dfp.delta.max())
    dfp = dfp[np.isclose(dfp.delta, dmax)]
    cfg = json.load(open(os.path.join(e3, "config.json")))
    p = model.ModelParams.from_run_config({"config": cfg["physics"]})
    z0 = 2.5758293035489004
    sig = np.logspace(np.log10(0.005), np.log10(0.04), 13)
    tt = np.logspace(0, 8, 17)
    n90 = np.full((len(sig), len(tt)), np.nan)
    for i, sg in enumerate(sig):
        _, sd0 = model.r_moments("b", 8, 0.0, float(sg), p, True, lam_scale=1)
        for j, t in enumerate(tt):
            L = float(np.log(t))
            _, sd = model.r_moments("b", 8, L, float(sg), p, True, lam_scale=1)
            dv, dw = model.fault_dm_nodes("b", "soft_abs", dmax, L,
                                          float(sg), p)
            p1 = model.predict_p1("b", 8, L, float(sg), p, True,
                                  z0 * sd0 * (sd / sd0), dv, dw, lam_scale=1)
            n90[i, j] = np.log(0.1) / np.log(1 - min(max(p1, 1e-9), 0.999))
    cl = bx.contour(tt, sig, n90, levels=[20], colors="k", linewidths=1.1)
    bx.clabel(cl, fmt={20: r"$N^{*}{=}20$"}, fontsize=5.5)
    c2 = bx.contour(tt, sig, n90, levels=[100], colors="0.4",
                    linewidths=0.8, linestyles="--")
    bx.clabel(c2, fmt={100: r"$100$"}, fontsize=5.5)
    for _, r in dfp.iterrows():
        n = np.log(0.1) / np.log(1 - min(max(r.p1_sim, 1e-9), 0.999))
        bx.plot(r.t_rel, r.sigma_nu, "o" if n <= 20 else "x", color="k",
                ms=2.8 if n <= 20 else 2.4, mfc="none")
    bx.set_xscale("log"); bx.set_yscale("log")
    bx.set_xlabel(r"$t/t_0$", fontsize=small["labelsize"])
    bx.set_ylabel(r"$\sigma_\nu$", fontsize=small["labelsize"])
    bx.tick_params(labelsize=small["ticksize"])
    bx.set_title("(b)", fontsize=7)
    _save(fig, out, "fig_val_n90")


# ---------------------------------------------------------------- E1 figure
def fig2_residual_law(e1, out):
    """Fig. 2: the two fault-free residual components (E1 formal record).

    Slice: phase A, m = 16, ratio readout, the four dispersion values.
    (a) sigma_r of encoding b leaves its calibration floor and grows as
    ln(t/t0). Note that sigma_r is NOT proportional to sigma_nu in this
    window (sigma_r/sigma_nu runs 85 down to 25 at t = 1e8), so the caption
    claims only the ln(t/t0) growth, which the E1 slope ranking supports.
    (b) mu_r for encodings a and b: the bias mapping carries a
    deterministic, nearly sigma_nu-independent offset; the differential
    pair does not.
    """
    df = pd.read_csv(os.path.join(e1, "summary.csv"))
    df = df[(df.phase == "A") & (df.m == 16) & (df.ratio_norm == 1)]
    sig = sorted(df.sigma_nu.unique())
    cmap = plt.get_cmap("viridis")
    cols = {s: cmap(0.06 + 0.80 * i / max(len(sig) - 1, 1))
            for i, s in enumerate(sig)}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(3.45, 1.20))

    marks = ["o", "s", "^", "D"]
    for i, s in enumerate(sig):
        d = df[(df.encoding == "b") & (df.sigma_nu == s)].sort_values("t_rel")
        ax.plot(d.t_rel, d.sigma_r, "-", marker=marks[i], lw=0.8, ms=2.0,
                mew=0.0, color=cols[s], label=r"$\sigma_\nu = %g$" % s)
    # A log x with a linear y is linear in ln(t/t0): the growth law of the
    # caption is then a straight line, and the series separate by slope.
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xticks([1e0, 1e2, 1e4, 1e6, 1e8])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel(r"$t/t_0$")
    ax.set_ylabel(r"$\sigma_r$")
    ax.set_title("(a) dispersion, encoding b", fontsize=7)

    for i, s in enumerate(sig):
        for enc, ls in [("a", "-"), ("b", "--")]:
            d = df[(df.encoding == enc) &
                   (df.sigma_nu == s)].sort_values("t_rel")
            bx.plot(d.t_rel, d.mean_r, ls, lw=0.8, marker=marks[i], ms=1.8,
                    mew=0.0, markevery=2, color=cols[s])
    bx.set_xscale("log")
    bx.set_ylim(-2.8, 0.62)
    bx.set_xticks([1e0, 1e2, 1e4, 1e6, 1e8])
    bx.set_yticks([-2, -1, 0])
    bx.set_xlabel(r"$t/t_0$")
    bx.set_ylabel(r"$\mu_r$")
    bx.set_title("(b) mean, encoding a vs b", fontsize=7)
    # Labelled by proximity rather than by a second legend or leader lines,
    # both of which covered data at this size.
    bx.text(1.3e6, 0.33, "encoding b", fontsize=5.4, ha="center")
    bx.text(1.3e6, -2.60, "encoding a", fontsize=5.4, ha="center")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=5.4, frameon=False, ncols=4,
               loc="upper center", bbox_to_anchor=(0.5, 1.02),
               columnspacing=1.0, handlelength=1.6, handletextpad=0.3,
               borderpad=0.0)
    fig.tight_layout(pad=0.12, w_pad=0.5, rect=(0, 0, 1, 0.90))
    _save(fig, out, "fig2_residual_law")


# ---------------------------------------------------------------- E5 figure
def fig7_transition(e5, out):
    """Fig. 7: the sign-flip confusability edge (E5 formal record).

    (a) Lawful per-cell flip probability C(t). (b) Per-group contamination
    against the share kappa * FPR0 that the budget allows it to consume;
    the analytic edge t_c is where the two meet, so both the threshold and
    the edge are drawn and the panel explains itself.

    Axis note: both panels are logarithmic and the record contains exact
    zeros (26 of the 68 simulation points saw no flip) plus model values as
    small as 1e-282 at early times. Those cannot appear on a log axis. The
    limits are set so that no POSITIVE simulation point is clipped: the
    smallest are 3.59e-8 in (a) and 7.81e-5 in (b).
    """
    df = pd.read_csv(os.path.join(e5, "flip_rate.csv"))
    rec = json.load(open(os.path.join(e5, "t_c.json")))
    tc, kappa = rec["t_c"], float(rec["kappa_fpr0"])
    sig = sorted(df.sigma_nu.unique())
    cmap = plt.get_cmap("viridis")
    cols = {s: cmap(0.06 + 0.80 * i / max(len(sig) - 1, 1))
            for i, s in enumerate(sig)}
    marks = ["o", "s", "^", "D"]
    assert df.pflip_sim[df.pflip_sim > 0].min() > 1e-8
    assert df.contam_sim[df.contam_sim > 0].min() > 5e-5
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(3.45, 1.20))

    for i, s in enumerate(sig):
        d = df[df.sigma_nu == s].sort_values("t_rel")
        ax.plot(d.t_rel, d.pflip_model, "-", lw=0.8, color=cols[s],
                label=r"$\sigma_\nu = %g$" % s)
        ax.plot(d.t_rel, d.pflip_sim, marks[i], ms=2.0, mfc="none", mew=0.7,
                color=cols[s], ls="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 0.4)
    ax.set_xticks([1e0, 1e2, 1e4, 1e6, 1e8])
    ax.set_yticks([1e-7, 1e-5, 1e-3, 1e-1])
    ax.set_xlabel(r"$t/t_0$")
    ax.set_ylabel(r"$C(t)$ per cell")
    ax.set_title("(a) lawful flip rate", fontsize=7)

    for i, s in enumerate(sig):
        d = df[df.sigma_nu == s].sort_values("t_rel")
        bx.plot(d.t_rel, d.contam_model, "-", lw=0.8, color=cols[s])
        bx.plot(d.t_rel, d.contam_sim, marks[i], ms=2.0, mfc="none", mew=0.7,
                color=cols[s], ls="none")
        t = tc.get("%g" % s, [None])[0]
        if t is not None and np.isfinite(t):
            bx.axvline(t, color=cols[s], lw=0.7, ls=":")
    bx.axhline(kappa, color="k", lw=0.8, ls="--")
    bx.text(1.4, kappa * 1.7, r"$\kappa\,\mathrm{FPR}_0$", fontsize=5.2,
            va="bottom")
    bx.set_xscale("log")
    bx.set_yscale("log")
    bx.set_ylim(5e-5, 1.8)
    bx.set_xticks([1e0, 1e2, 1e4, 1e6, 1e8])
    bx.set_yticks([1e-4, 1e-2, 1e0])
    bx.set_xlabel(r"$t/t_0$")
    bx.set_ylabel("contamination")
    bx.set_title(r"(b) group contamination", fontsize=7)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=5.4, frameon=False, ncols=4,
               loc="upper center", bbox_to_anchor=(0.5, 1.02),
               columnspacing=1.0, handlelength=1.6, handletextpad=0.3,
               borderpad=0.0)
    fig.tight_layout(pad=0.12, w_pad=0.5, rect=(0, 0, 1, 0.90))
    _save(fig, out, "fig7_transition")


# ---------------------------------------------------------------- E6 figure
def fig3_checker(e6, out):
    """Fig. 3: the digitized checker holds its error budget (E6 record).

    (a) Relative band error of the 32-segment piecewise-linear evaluator
    over ln t, eps_f = |f_hw - f_exact| / f_exact. The axis is linear
    because eps_f spans 0 to 1.78 percent against a 6.7 percent budget: the
    margin is a distance the reader should be able to read off, and a log
    axis four decades tall for data confined under two percent wastes the
    panel.
    (b) Bit-true deployment at b_y = 8. Only this width is plotted because
    it is the only one recorded with the piecewise-linear evaluator in the
    loop (mode "pwl"); b_y = 6 and 10 were recorded in "register" mode and
    are therefore not the same experiment. The float reference is the same
    band evaluated in floating point; the control keeps the analog growth
    law after the ADC and collapses the rate.
    """
    e = pd.read_csv(os.path.join(e6, "eps_curve.csv"))
    b = pd.read_csv(os.path.join(e6, "fpr_bittrue.csv"))
    b8 = b[(b.by == 8) & (b["mode"] == "pwl")].sort_values("t_rel")
    assert len(b8) == 9, "expected the nine time points of the b_y = 8 record"
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(3.45, 1.20))

    ax.plot(e.t_rel, 100 * e.eps_f, "-", lw=0.6, color="#1f77b4")
    ax.axhline(6.7, color="k", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_ylim(0, 7.6)
    ax.set_xticks([1e0, 1e2, 1e4, 1e6, 1e8])
    ax.set_yticks([0, 2, 4, 6])
    ax.text(3e3, 6.35, r"budget $6.7\%$", fontsize=5.4, va="top",
            ha="center")
    pk = e.loc[e.eps_f.idxmax()]
    ax.annotate(r"max $1.78\%$", xy=(pk.t_rel, 100 * pk.eps_f),
                xytext=(2e2, 3.4), fontsize=5.4,
                arrowprops=dict(arrowstyle="->", lw=0.5, color="0.35",
                                shrinkA=1, shrinkB=1))
    ax.set_xlabel(r"$t/t_0$")
    ax.set_ylabel(r"$\varepsilon_f$ (%)")
    ax.set_title("(a) evaluator error", fontsize=7)

    bx.plot(b8.t_rel, b8.fpr_float_ref, "-", lw=2.0, color="0.72",
            solid_capstyle="round", label="float reference")
    bx.plot(b8.t_rel, b8.fpr_bittrue, "-o", lw=0.8, ms=2.4, mfc="none",
            mew=0.8, color="#1f77b4", label="bit true")
    bx.plot(b8.t_rel, b8.fpr_wrong_rho, "-.^", lw=0.8, ms=2.4, mew=0.0,
            color="#d62728", label=r"analog $\rho$")
    bx.set_xscale("log")
    bx.set_yscale("log")
    bx.set_ylim(2e-5, 3e-1)
    bx.set_xticks([1e0, 1e2, 1e4, 1e6, 1e8])
    bx.set_yticks([1e-4, 1e-3, 1e-2])
    bx.set_xlabel(r"$t/t_0$")
    bx.set_ylabel("false-positive rate")
    bx.set_title(r"(b) deployment at $b_y = 8$", fontsize=7)
    # Legend in the headroom above the data; every lower region is occupied.
    bx.legend(fontsize=5.2, frameon=False, loc="upper left", ncols=2,
              columnspacing=0.7, handlelength=1.5, handletextpad=0.3,
              labelspacing=0.12, borderpad=0.0)
    fig.tight_layout(pad=0.12, w_pad=0.5)
    _save(fig, out, "fig3_checker")


FIGURES = {                      # figure name -> (builder, required records)
    "fig1_construction": (lambda a: fig1_construction(a.out), ()),
    "fig2_residual_law": (lambda a: fig2_residual_law(a.e1, a.out), ("e1",)),
    "fig3_checker":      (lambda a: fig3_checker(a.e6, a.out), ("e6",)),
    "fig4_validation":   (lambda a: fig4_validation(a.e2, a.e3, a.out),
                          ("e2", "e3")),
    "fig5_n90":          (lambda a: fig5_n90(a.e3, a.out), ("e3",)),
    "fig6_motivation":   (lambda a: fig6_motivation(a.e4, a.out), ("e4",)),
    "fig7_transition":   (lambda a: fig7_transition(a.e5, a.out), ("e5",)),
    "fig_val_n90":       (lambda a: fig_val_n90(a.e2, a.e3, a.out),
                          ("e2", "e3")),   # retained, not placed in the brief
}
PAPER_ORDER = ["fig1_construction", "fig2_residual_law", "fig3_checker",
               "fig4_validation", "fig5_n90", "fig6_motivation",
               "fig7_transition"]


def main():
    ap = argparse.ArgumentParser(
        description="Build the PCM_DriftCheck brief figures. Record "
                    "directories are required only for the figures selected; "
                    "fig1_construction is data free.")
    for k in ("e1", "e2", "e3", "e4", "e5", "e6"):
        ap.add_argument("--" + k, default=None,
                        help="path to the formal %s record directory" % k.upper())
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)),
                    help="output directory (default: this script's directory)")
    ap.add_argument("--only", nargs="+", default=None, choices=sorted(FIGURES),
                    help="build only these figures (default: the seven in the "
                         "brief)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    plt.rcParams.update(RC)
    for name in (a.only if a.only else PAPER_ORDER):
        build, needs = FIGURES[name]
        missing = [k for k in needs if getattr(a, k) is None]
        if missing:
            print("skip %s: needs --%s" % (name, ", --".join(missing)))
            continue
        build(a)
    return


if __name__ == "__main__":
    main()
