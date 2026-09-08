#!/usr/bin/env python3
"""E4: three-detector baseline comparison on seed-identical E3 ensembles.

Implements docs/E4_Design_Spec_2026-08-30.md (frozen). Detectors:

  A(b): exact equality of b-bit quantized outputs, round(y_ck/q) vs
        round(lam * sum_j y_j / q), q = FS / 2^b with FS spanning the t0
        healthy outputs of all logical columns (favorable calibration);
  B:    constant tolerance |r| > theta0 (t0 empirical 1% quantile);
  C:    drift-aware band |r| > theta0 * rho_model(t)  (this work; must
        reproduce the E3 official record bit for bit, spec gate Q3).

Run from the project root:

    python experiments/E4_baseline_comparison.py            (full)
    python experiments/E4_baseline_comparison.py --smoke    (pipeline check)

Outputs per run: results/E4/<runstamp>/{config.json, fpr_compare.csv,
discrimination.csv}, figures/E4/*_<runstamp>.{pdf,png}, logs/E4_<runstamp>.log.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "src"))

import pcm_drift  # noqa: E402
from pcm_drift import checksum, encodings, io_utils, model  # noqa: E402

sys.path.insert(0, str(_HERE.parent))
from E1_config import E1Config  # noqa: E402
from E3_config import E3Config  # noqa: E402
from E4_config import E4Config  # noqa: E402
from E3_validity_domain_scan import ENC_ID, build_ensemble, physics_params  # noqa: E402

FPR_FIELDS = ["encoding", "sigma_nu", "m", "c", "ratio_norm", "t_rel",
              "detector", "threshold", "fpr"]
DISC_FIELDS = ["encoding", "sigma_nu", "m", "c", "ratio_norm", "t_rel", "delta",
               "detector", "p1", "fpr", "delta_p", "n_trials"]


def pools_with_yck(arrays, t_grid, cfg):
    """OFF-side residual and checksum-output pools plus per-(t, array) scales."""
    A = len(arrays)
    K = arrays[0][1].K
    C = arrays[0][1].C
    N = arrays[0][2].shape[0]
    r_off = np.empty((len(t_grid), A, N, K), dtype=np.float32)
    yck_off = np.empty((len(t_grid), A, N, K), dtype=np.float32)
    s_arr = np.empty((len(t_grid), A), dtype=np.float64)
    for a, (state, layout, X) in enumerate(arrays):
        for ti, t in enumerate(t_grid):
            M, _ = encodings.effective_matrix(state, layout, float(t), False, cfg)
            Y = X @ M
            r_off[ti, a] = checksum.residuals(Y, layout).astype(np.float32)
            yck_off[ti, a] = Y[:, C:C + K].astype(np.float32)
            s_arr[ti, a] = encodings._ratio_scale(state, float(t), True)
    return r_off, yck_off, s_arr


def detector_flags(kind, r, yck, s, q, thr, norm_on):
    """Boolean alarm array for one detector on (r, yck) samples.

    kind 'A': codes differ after quantization of both compared quantities in
    the readout domain actually digitized (ON: s scales both sides equally).
    kind 'B'/'C': |r_readout| > thr with r_readout = s*r (ON) or r (OFF).
    """
    if kind == "A":
        y1 = (s * yck) if norm_on else yck
        y2 = y1 - ((s * r) if norm_on else r)
        return np.round(y1 / q) != np.round(y2 / q)
    stat = np.abs((s * r) if norm_on else r)
    return stat > thr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results/E4")
    args = ap.parse_args()

    cfg = E1Config()
    e3 = E3Config()
    e4 = E4Config()
    p = physics_params(cfg)
    t_grid = list(e3.t_rel_grid)
    sig_on = list(e4.sigma_nu_on)
    n_trials = e4.n_trials
    if args.smoke:
        sig_on = [0.02]
        t_grid = [1.0, 1e2, 1e8]
        e3 = dataclasses.replace(e3, n_arrays=4, n_mvm=300)
        n_trials = 600

    paths = io_utils.resolve_paths(args.out, exp="E4", suffix="_smoke" if args.smoke else "")
    log = io_utils.get_logger("E4", paths.logs / f"E4_{paths.runstamp}.log")
    t_start = time.perf_counter()
    log.info("E4 run %s starting (smoke=%s)", paths.runstamp, args.smoke)
    io_utils.save_json(
        io_utils.config_snapshot(e4, vars(args), {
            "runstamp": paths.runstamp, "physics": dataclasses.asdict(cfg),
            "e3_seed_coords": {"seed_base": e3.seed_base, "n_arrays": e3.n_arrays,
                               "n_mvm": e3.n_mvm},
            "pcm_drift_version": pcm_drift.__version__,
        }),
        paths.run_dir / "config.json",
    )

    confs = [(s, True) for s in sig_on]
    if not args.smoke:
        confs.append((e4.sigma_off_slice, False))
    m, c = e4.m, e4.c
    mi_g = e3.m_grid.index(m)
    ci_g = e3.c_grid.index(c)
    lam = c / m
    rows_fpr, rows_disc = [], []

    for sig, norm_on in confs:
        tc = time.perf_counter()
        si_g = e3.sigma_nu_grid.index(sig)
        arrays = build_ensemble("b", si_g, mi_g, ci_g, cfg, e3)
        r_off, yck_off, s_arr = pools_with_yck(arrays, t_grid, cfg)
        A = len(arrays)
        N = arrays[0][2].shape[0]

        # calibrations at t0
        s0 = s_arr[0][:, None, None] if norm_on else 1.0
        theta0 = float(np.quantile(np.abs(s0 * r_off[0].astype(np.float64)), 1.0 - e4.fpr0))
        _, sd_t0 = model.r_moments("b", m, 0.0, sig, p, norm_on, lam_scale=c)
        # baseline A full scale: span of ALL logical column outputs at t0, in
        # the digitized (readout) domain; recomputed per array batch cheaply
        y_all_max = 0.0
        for a, (state, layout, X) in enumerate(arrays):
            M0, _ = encodings.effective_matrix(state, layout, 1.0, False, cfg)
            Y0 = X @ M0
            scale0 = s_arr[0, a] if norm_on else 1.0
            y_all_max = max(y_all_max, float(np.abs(scale0 * Y0).max()))
        FS = 2.0 * y_all_max
        q_of = {b: FS / (2 ** b) for b in e4.adc_bits}
        log.info("config sigma=%.3g norm=%s: theta0=%.5f FS=%.2f q=%s",
                 sig, "on" if norm_on else "off", theta0,
                 FS, {b: round(qq, 4) for b, qq in q_of.items()})

        detectors = [(f"A{b}", ("A", q_of[b])) for b in e4.adc_bits]
        detectors += [("B_const", ("T", theta0))]

        frng = np.random.default_rng(np.random.SeedSequence(
            [e3.seed_base, e4.fault_seed_branch, ENC_ID["b"], si_g, mi_g, ci_g, int(norm_on)]))

        for ti, t in enumerate(t_grid):
            L = float(np.log(t))
            _, sd_t = model.r_moments("b", m, L, sig, p, norm_on, lam_scale=c)
            f_band = theta0 * (sd_t / sd_t0)
            dets = detectors + [("C_band", ("T", f_band))]
            r_t = r_off[ti].astype(np.float64)
            y_t = yck_off[ti].astype(np.float64)
            s_t = s_arr[ti][:, None, None]
            for name, (kind, par) in dets:
                if kind == "A":
                    fl = detector_flags("A", r_t, y_t, s_t, par, None, norm_on)
                else:
                    fl = detector_flags("B", r_t, y_t, s_t, None, par, norm_on)
                fpr = float(fl.mean())
                rows_fpr.append({"encoding": "b", "sigma_nu": sig, "m": m, "c": c,
                                 "ratio_norm": int(norm_on), "t_rel": t,
                                 "detector": name, "threshold": par, "fpr": fpr})

            # discrimination probes
            if t in e4.probe_t:
                for d in e4.probe_deltas:
                    aa = frng.integers(0, A, n_trials)
                    rows_i = frng.integers(0, cfg.R, n_trials)
                    cols = frng.integers(0, cfg.C, n_trials)
                    nn = frng.integers(0, N, n_trials)
                    br = frng.integers(0, 2, n_trials)
                    sg = np.where(frng.integers(0, 2, n_trials) == 0, 1.0, -1.0)
                    hits = {name: 0 for name, _ in dets}
                    for a in range(A):
                        sel = aa == a
                        if not sel.any():
                            continue
                        state, layout, X = arrays[a]
                        i = rows_i[sel]
                        j = cols[sel]
                        n_in = nn[sel]
                        pcol = np.where(br[sel] == 0, layout.pos_cols[j], layout.neg_cols[j])
                        bsign = np.where(br[sel] == 0, 1.0, -1.0)
                        g0 = state.g0[i, pcol]
                        nu = state.nu[i, pcol]
                        g_new = np.maximum(g0 + sg[sel] * d, cfg.g0_floor)
                        dm = bsign * (g_new - g0) * np.exp(-nu * L)
                        shift = -lam * X[n_in, i] * dm
                        k = j // layout.m
                        r1 = r_off[ti, a, n_in, k].astype(np.float64) + shift
                        y1 = yck_off[ti, a, n_in, k].astype(np.float64)
                        sv = s_arr[ti, a]
                        for name, (kind, par) in dets:
                            if kind == "A":
                                fla = detector_flags("A", r1, y1, sv, par, None, norm_on)
                            else:
                                fla = detector_flags("B", r1, y1, sv, None, par, norm_on)
                            hits[name] += int(fla.sum())
                    fpr_by = {r["detector"]: r["fpr"] for r in rows_fpr
                              if r["sigma_nu"] == sig and r["ratio_norm"] == int(norm_on)
                              and r["t_rel"] == t}
                    for name, _ in dets:
                        p1 = hits[name] / n_trials
                        rows_disc.append({"encoding": "b", "sigma_nu": sig, "m": m, "c": c,
                                          "ratio_norm": int(norm_on), "t_rel": t, "delta": d,
                                          "detector": name, "p1": p1, "fpr": fpr_by[name],
                                          "delta_p": p1 - fpr_by[name], "n_trials": n_trials})
        log.info("config sigma=%.3g norm=%s done in %.1f s",
                 sig, "on" if norm_on else "off", time.perf_counter() - tc)

    for name, rows, fields in (("fpr_compare.csv", rows_fpr, FPR_FIELDS),
                               ("discrimination.csv", rows_disc, DISC_FIELDS)):
        with open(paths.run_dir / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    log.info("written: fpr_compare.csv (%d rows), discrimination.csv (%d rows)",
             len(rows_fpr), len(rows_disc))

    # spec verdict lines
    for sig, norm_on in confs:
        sub = {r["detector"]: r["fpr"] for r in rows_fpr
               if r["sigma_nu"] == sig and r["ratio_norm"] == int(norm_on)
               and r["t_rel"] == (t_grid[1] if len(t_grid) > 1 else t_grid[0])}
        log.info("one-decade FPR sigma=%.3g norm=%s: %s", sig, "on" if norm_on else "off",
                 {k: round(v, 4) for k, v in sorted(sub.items())})

    make_figures(rows_fpr, rows_disc, t_grid, sig_on, e4, paths, log, args.smoke)
    log.info("E4 run %s complete in %.1f s; every number above traces to %s",
             paths.runstamp, time.perf_counter() - t_start, paths.run_dir)
    return 0


def make_figures(rows_fpr, rows_disc, t_grid, sig_on, e4, paths, log, smoke):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {"A6": ("tab:red", ":"), "A8": ("tab:red", "--"), "A10": ("tab:red", "-."),
              "B_const": ("tab:orange", "-"), "C_band": ("tab:blue", "-")}
    panels = [(s, 1) for s in sig_on] + ([(e4.sigma_off_slice, 0)] if not smoke else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.2),
                             squeeze=False, sharey=True)
    for ax, (sig, nm) in zip(axes[0], panels):
        for det, (col, ls) in styles.items():
            pts = sorted((r["t_rel"], max(r["fpr"], 5e-4)) for r in rows_fpr
                         if r["sigma_nu"] == sig and r["ratio_norm"] == nm and r["detector"] == det)
            if pts:
                ax.plot([q[0] for q in pts], [q[1] for q in pts], color=col, ls=ls,
                        lw=1.5 if det == "C_band" else 1.1, label=det)
        ax.axhspan(0.005, 0.02, color="tab:blue", alpha=0.08)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(5e-4, 1.3)
        ax.set_xlabel("$t/t_0$")
        ax.set_title(f"$\\sigma_\\nu$={sig}, norm {'on' if nm else 'off'}", fontsize=9)
    axes[0][0].set_ylabel("false-alarm rate per MVM")
    axes[0][0].legend(fontsize=6.5)
    fig.suptitle("E4: exact-equality family (A, red) vs constant tolerance (B) vs drift-aware band (C)",
                 fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(paths.figures / f"fig_e4_fpr_baselines_{paths.runstamp}.{ext}", dpi=200)
    plt.close(fig)

    if rows_disc:
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        dets = ["A6", "A8", "A10", "B_const", "C_band"]
        probes = sorted({(r["t_rel"], r["delta"]) for r in rows_disc
                         if r["sigma_nu"] == 0.02 and r["ratio_norm"] == 1})
        width = 0.15
        for di, det in enumerate(dets):
            vals = []
            for t, d in probes:
                v = [r["delta_p"] for r in rows_disc
                     if r["sigma_nu"] == 0.02 and r["ratio_norm"] == 1
                     and r["t_rel"] == t and r["delta"] == d and r["detector"] == det]
                vals.append(v[0] if v else 0.0)
            xs = np.arange(len(probes)) + (di - 2) * width
            ax.bar(xs, vals, width, label=det,
                   color=styles[det][0], alpha=0.5 + 0.5 * (det == "C_band"))
        ax.set_xticks(np.arange(len(probes)))
        ax.set_xticklabels([f"t=1e{int(np.log10(t))}\n$\\delta$={d}" for t, d in probes], fontsize=7)
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_ylabel("discrimination margin  $p_1$ - FPR")
        ax.set_title("$\\sigma_\\nu$=0.02, norm on", fontsize=9)
        ax.legend(fontsize=6.5)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(paths.figures / f"fig_e4_discrimination_{paths.runstamp}.{ext}", dpi=200)
        plt.close(fig)
    log.info("figures written to %s", paths.figures)


if __name__ == "__main__":
    sys.exit(main())
