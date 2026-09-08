#!/usr/bin/env python3
"""E2: analytical residual model vs stored E1 samples (moment validation).

Implements the validation protocol of docs/E2_Derivation_2026-08-29.md,
section 9: for encodings a and b, compare the model's (mu_r, sigma_r, FPR)
against the empirical values recomputed from an E1 run's residuals_<enc>.npz,
over the full (sigma_nu x m x ratio_norm x t) grid, with no fitted parameters.
Physical parameters are read from the source run's config.json; thresholds are
the source run's calibrated theta (calibration inputs, not fits).

Run from the project root:

    python experiments/E2_residual_model_validation.py --run results/E1/<runstamp>

Outputs per run: results/E2/<runstamp>/{config.json, model_vs_data.csv,
band_ratio.csv}, figures/E2/*_<runstamp>.{pdf,png}, logs/E2_<runstamp>.log.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "src"))

import pcm_drift  # noqa: E402
from pcm_drift import io_utils, model  # noqa: E402

sys.path.insert(0, str(_HERE.parent))
from E2_config import E2Config  # noqa: E402

FIELDS = [
    "encoding", "sigma_nu", "m", "ratio_norm", "t_rel",
    "mu_data", "mu_model", "sigma_data", "sigma_model", "sigma_rel_err", "sigma_model_cond",
    "fpr_data", "fpr_model", "fpr_abs_err", "theta", "n", "flag",
]


def load_thresholds(run_dir: Path) -> dict:
    th = {}
    with open(run_dir / "thresholds.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            th[(row["encoding"], row["sigma_nu"], row["m"], row["ratio_norm"])] = float(row["theta"])
    return th


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="source E1 runstamp directory (contains residuals_<enc>.npz)")
    ap.add_argument("--enc", nargs="+", choices=["a", "b"], default=None,
                    help="encodings to validate (default: E2Config.encodings)")
    ap.add_argument("--out", default="results/E2", help="results root (default: results/E2)")
    args = ap.parse_args()

    e2 = E2Config()
    encs = tuple(args.enc) if args.enc else e2.encodings
    run_dir = Path(args.run)
    src_cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    p = model.ModelParams.from_run_config(src_cfg)
    src_stamp = src_cfg.get("runstamp", run_dir.name)

    paths = io_utils.resolve_paths(args.out, exp="E2")
    log = io_utils.get_logger("E2", paths.logs / f"E2_{paths.runstamp}.log")
    t_start = time.perf_counter()
    log.info("E2 run %s starting; source E1 run: %s (%s)", paths.runstamp, src_stamp, run_dir)
    io_utils.save_json(
        io_utils.config_snapshot(
            e2, vars(args),
            {"runstamp": paths.runstamp, "source_run": str(run_dir), "source_runstamp": src_stamp,
             "model_params": asdict(p), "pcm_drift_version": pcm_drift.__version__},
        ),
        paths.run_dir / "config.json",
    )

    st = model.mc_selftest(p, np.random.default_rng(e2.mc_seed), e2.mc_n)
    if st["worst_rel_err"] > e2.mc_tol_rel:
        raise AssertionError(f"model MC selftest FAILED: worst rel err {st['worst_rel_err']:.3e}")
    log.info("model MC selftest PASS: worst cell-moment rel err = %.2e (tol %.2e)",
             st["worst_rel_err"], e2.mc_tol_rel)

    sig_grid = [str(s) for s in src_cfg["config"]["sigma_nu_grid"]]
    m_grid = src_cfg["config"]["m_grid"]
    t_grid = np.asarray(src_cfg["config"]["t_rel_grid"], dtype=float)
    thresholds = load_thresholds(run_dir)

    rows, n_warn = [], 0
    for enc in encs:
        t_enc = time.perf_counter()
        z = np.load(run_dir / f"residuals_{enc}.npz")
        for si, sig_s in enumerate(sig_grid):
            sig = float(sig_s)
            for m in m_grid:
                pair = {nm: z[f"r_s{si}_m{m}_norm{nm}"] for nm in (0, 1)}  # [T,A,N,K]
                # realized per-array ratio scales s_a at t=1e4 (exact relation
                # r_on = s r_off); their sample moments separate the nu_ref
                # 20-draw sampling effect from model error in the norm-on
                # comparison (doc sec. 9, P5)
                with np.errstate(divide="ignore", invalid="ignore"):
                    rat = pair[1][4].astype(np.float64) / pair[0][4].astype(np.float64)
                s_a = np.nanmedian(rat, axis=(1, 2))
                for nm in (0, 1):
                    theta = thresholds[(enc, sig_s, str(m), str(nm))]
                    r_all = pair[nm]
                    for ti, t in enumerate(t_grid):
                        r = r_all[ti].astype(np.float64)
                        mu_d = float(r.mean())
                        sd_d = float(r.std())
                        fpr_d = float((np.abs(r) > theta).mean())
                        L = float(np.log(t))
                        mu_m, sd_m = model.r_moments(enc, m, L, sig, p, bool(nm))
                        fpr_m = model.predict_fpr(mu_m, sd_m, theta)
                        if nm == 1:
                            mu0, sd0 = model.r_moments(enc, m, L, sig, p, False)
                            sL = np.power(np.maximum(s_a, 1e-12), L / np.log(t_grid[4]))
                            es1, es2 = float(np.mean(sL)), float(np.mean(sL**2))
                            sd_cond = float(np.sqrt(max(es2 * (sd0**2 + mu0**2) - (es1 * mu0) ** 2, 0.0)))
                        else:
                            sd_cond = ""
                        s_err = sd_m / sd_d - 1.0 if sd_d > 0 else float("nan")
                        f_err = fpr_m - fpr_d
                        flags = []
                        if abs(s_err) > e2.tol_sigma_rel:
                            flags.append("SIGMA")
                        if abs(f_err) > e2.tol_fpr_abs:
                            flags.append("FPR")
                        if enc == "a" and abs(mu_d) > e2.mu_floor and abs(mu_m / mu_d - 1.0) > e2.tol_mu_rel:
                            flags.append("MU")
                        n_warn += bool(flags)
                        rows.append({
                            "encoding": enc, "sigma_nu": sig, "m": m, "ratio_norm": nm, "t_rel": t,
                            "mu_data": mu_d, "mu_model": mu_m,
                            "sigma_data": sd_d, "sigma_model": sd_m, "sigma_rel_err": s_err,
                            "sigma_model_cond": sd_cond,
                            "fpr_data": fpr_d, "fpr_model": fpr_m, "fpr_abs_err": f_err,
                            "theta": theta, "n": r.size, "flag": "+".join(flags),
                        })
            log.info("enc=%s sigma_nu=%s done", enc, sig_s)
        log.info("enc=%s finished in %.1f s", enc, time.perf_counter() - t_enc)

    with open(paths.run_dir / "model_vs_data.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    log.info("written: model_vs_data.csv (%d rows, %d flagged)", len(rows), n_warn)

    # aggregate verdicts (pre-registered tolerances, doc sec. 9)
    for enc in encs:
        er = [abs(r["sigma_rel_err"]) for r in rows if r["encoding"] == enc]
        ef = [abs(r["fpr_abs_err"]) for r in rows if r["encoding"] == enc]
        log.info("verdict enc=%s: |sigma rel err| median %.3f / p90 %.3f / max %.3f ; "
                 "|FPR err| median %.3f / p90 %.3f / max %.3f",
                 enc, np.median(er), np.percentile(er, 90), max(er),
                 np.median(ef), np.percentile(ef, 90), max(ef))

    # band ratio table rho(t) = sigma_model(t)/sigma_model(t0), norm on (doc sec. 7)
    with open(paths.run_dir / "band_ratio.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["encoding", "sigma_nu", "m", "t_rel", "rho"])
        for enc in encs:
            for sig_s in sig_grid:
                sig = float(sig_s)
                base = None
                for t in t_grid:
                    _, sd = model.r_moments(enc, e2.fig_m, float(np.log(t)), sig, p, True)
                    base = sd if base is None else base
                    w.writerow([enc, sig, e2.fig_m, t, sd / base])
    log.info("written: band_ratio.csv (norm on, m=%d)", e2.fig_m)

    make_figures(rows, t_grid, sig_grid, e2, paths, log, encs)
    log.info("E2 run %s complete in %.1f s; every number above traces to %s",
             paths.runstamp, time.perf_counter() - t_start, paths.run_dir)
    return 0


def make_figures(rows, t_grid, sig_grid, e2: E2Config, paths, log, encs) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def slice_rows(enc, sig, nm, key):
        pts = sorted((r["t_rel"], r[key]) for r in rows
                     if r["encoding"] == enc and r["sigma_nu"] == float(sig)
                     and r["m"] == e2.fig_m and r["ratio_norm"] == nm)
        return [p[0] for p in pts], [p[1] for p in pts]

    if "b" in encs:
        fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4), sharex=True)
        for col, key_d, key_m, ylab, ylog in (
            (0, "sigma_data", "sigma_model", r"$\sigma_r$", True),
            (1, "fpr_data", "fpr_model", "FPR", True),
        ):
            for row_i, nm in enumerate((0, 1)):
                ax = axes[row_i][col]
                for sig in sig_grid:
                    t, yd = slice_rows("b", sig, nm, key_d)
                    _, ym = slice_rows("b", sig, nm, key_m)
                    if ylog:
                        yd = [max(v, 1e-4) for v in yd]
                        ym = [max(v, 1e-4) for v in ym]
                    ln, = ax.plot(t, yd, marker="o", ms=3, lw=0, label=f"data $\\sigma_\\nu$={sig}")
                    ax.plot(t, ym, lw=1.2, color=ln.get_color())
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title(f"enc b, ratio_norm={'on' if nm else 'off'}", fontsize=9)
                if row_i == 1:
                    ax.set_xlabel("$t/t_0$")
                if col == 0:
                    ax.set_ylabel(ylab)
        axes[0][0].legend(fontsize=6)
        fig.suptitle("E2: model (lines) vs E1 data (dots), encoding b (m=16)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        for ext in ("pdf", "png"):
            fig.savefig(paths.figures / f"fig_e2_model_vs_data_b_{paths.runstamp}.{ext}", dpi=200)
        plt.close(fig)
        log.info("figure written: fig_e2_model_vs_data_b_%s.pdf/.png", paths.runstamp)

    if "a" in encs:
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharex=True)
        for ax, nm in zip(axes, (0, 1)):
            for sig in sig_grid:
                t, yd = slice_rows("a", sig, nm, "mu_data")
                _, ym = slice_rows("a", sig, nm, "mu_model")
                ln, = ax.plot(t, yd, marker="o", ms=3, lw=0, label=f"data $\\sigma_\\nu$={sig}")
                ax.plot(t, ym, lw=1.2, color=ln.get_color())
            ax.set_xscale("log")
            ax.set_title(f"enc a, ratio_norm={'on' if nm else 'off'}", fontsize=9)
            ax.set_xlabel("$t/t_0$")
        axes[0].set_ylabel(r"$\mu_r$")
        axes[0].legend(fontsize=6)
        fig.suptitle("E2: deterministic mean bias, model (lines) vs data (dots), encoding a (m=16)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        for ext in ("pdf", "png"):
            fig.savefig(paths.figures / f"fig_e2_mean_bias_a_{paths.runstamp}.{ext}", dpi=200)
        plt.close(fig)
        log.info("figure written: fig_e2_mean_bias_a_%s.pdf/.png", paths.runstamp)


if __name__ == "__main__":
    sys.exit(main())
