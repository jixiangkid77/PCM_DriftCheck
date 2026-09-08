#!/usr/bin/env python3
"""E1: naive checksum failure demonstration (encoding as the primary variable).

Master dossier section 6, row E1; implements docs/E1_Design_Spec_2026-08-22.md.

Phase A (fault-free): false-positive rate of a constant threshold, calibrated
at t0 to FPR0, as a function of time under per-cell power-law drift; three
encodings (a bias, b differential, c sign-magnitude) x ratio normalization
off/on x sigma_nu x segment size m.

Phase B (single persistent fault applied at t0): detection rate of the same
constant threshold over time; sa0 / sa1 / soft deviation on one random data
cell of a monitored group (light version per spec section 5; full fault
coverage is E5's scope).

Run from the project root (D:\\Project\\PCM_DriftCheck, WSL
/mnt/d/Project/PCM_DriftCheck):

    python experiments/E1_naive_checksum_failure.py --smoke          # pipeline check
    python experiments/E1_naive_checksum_failure.py --phase A B      # full run

An invariant self-test (r(t0) = 0 with programming spread off, r nonzero under
drift) runs at startup and aborts on failure. Smoke runs write to runstamp
directories suffixed _smoke (results, figures, log) so verification runs are
never confusable with data runs. Outputs per run:
results/E1/<runstamp>/{config.json, thresholds.csv, residuals_<enc>.npz,
summary.csv, c2_slope_ranking.csv}, figures/E1/*_<runstamp>.{pdf,png},
logs/E1_<runstamp>.log.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "src"))

import pcm_drift  # noqa: E402
from pcm_drift import checksum, encodings, faults, io_utils, metrics  # noqa: E402
from pcm_drift.crossbar import matmul  # noqa: E402

sys.path.insert(0, str(_HERE.parent))
from E1_config import (  # noqa: E402
    E1Config,
    make_inputs,
    make_weights,
    rng_device,
    rng_fault,
    smoke_config,
)

SUMMARY_FIELDS = [
    "phase",
    "encoding",
    "sigma_nu",
    "m",
    "ratio_norm",
    "t_rel",
    "fault_kind",
    "d",
    "n",
    "n_exceed",
    "rate",
    "ci_lo",
    "ci_hi",
    "mean_r",
    "sigma_r",
    "q01",
    "q05",
    "q25",
    "q50",
    "q75",
    "q95",
    "q99",
    "p_flip_pcm_ref",
    "p_flip_fixed_raw",
    "p_flip_fixed_norm",
    "n_cond",
    "n_exceed_cond",
    "rate_cond",
]


def row(**kw) -> dict:
    r = {f: "" for f in SUMMARY_FIELDS}
    r.update(kw)
    return r


# ---------------------------------------------------------------------------
# startup invariant self-test
# ---------------------------------------------------------------------------


def invariant_selftest(cfg: E1Config, log, backend: str) -> None:
    """Abort unless r(t0) = 0 (spread off) and r(1e8) != 0 for all encodings.

    Verifies the checksum construction algebra of spec sections 2 and 4 on a
    small array before any data-producing run.
    """
    tiny = replace(cfg, R=16, C=16, m_grid=(4,), prog_spread=False)
    W = np.clip(np.random.default_rng(12345).normal(0.0, tiny.w_sigma, (tiny.R, tiny.C)), -1.0, 1.0)
    X = np.random.default_rng(67890).uniform(0.0, 1.0, (8, tiny.R))
    worst_t0, worst_drift = 0.0, np.inf
    for enc in ("a", "b", "c"):
        st, lay = encodings.build(enc, W, 4, 0.02, np.random.default_rng(42), tiny)
        for norm in (False, True):
            M0, _ = encodings.effective_matrix(st, lay, 1.0, norm, tiny)
            r0 = checksum.residuals(matmul(X, M0, backend), lay)
            worst_t0 = max(worst_t0, float(np.max(np.abs(r0))))
            M8, _ = encodings.effective_matrix(st, lay, 1e8, norm, tiny)
            r8 = checksum.residuals(matmul(X, M8, backend), lay)
            worst_drift = min(worst_drift, float(np.max(np.abs(r8))))
    if worst_t0 >= 1e-9:
        raise AssertionError(f"selftest FAILED: max |r(t0)| = {worst_t0:.3e} >= 1e-9")
    if worst_drift <= 1e-4:
        raise AssertionError(f"selftest FAILED: drifted residual too small ({worst_drift:.3e})")
    log.info(
        "selftest PASS: max |r(t0)| = %.3e (< 1e-9, spread off), min over cases of "
        "max |r(1e8)| = %.3e (> 1e-4)",
        worst_t0,
        worst_drift,
    )


# ---------------------------------------------------------------------------
# Phase A
# ---------------------------------------------------------------------------


def run_phase_a(cfg: E1Config, paths, log, backend: str):
    """Fault-free residual collection, threshold calibration, FPR(t) metrics.

    Returns (thetas, rows, slope_rows) where thetas maps
    (enc, si, mi, norm) -> theta.
    """
    t_grid = np.asarray(cfg.t_rel_grid, dtype=float)
    T, A, N = len(t_grid), cfg.n_arrays, cfg.n_mvm
    thetas: dict = {}
    rows: list = []
    slope_rows: list = []

    for enc in cfg.encodings:
        t_enc = time.perf_counter()
        npz_payload = {
            "t_rel_grid": t_grid,
            "sigma_nu_grid": np.asarray(cfg.sigma_nu_grid),
            "m_grid": np.asarray(cfg.m_grid),
        }
        for si, sig in enumerate(cfg.sigma_nu_grid):
            for mi, m in enumerate(cfg.m_grid):
                K = cfg.C // m
                R_all = {nm: np.empty((T, A, N, K), dtype=np.float32) for nm in (0, 1)}
                FLIP = np.zeros((T, A, K, 2), dtype=bool) if enc == "c" else None
                PFLIP = np.zeros((T, A, 3), dtype=float) if enc == "c" else None

                for a_i in range(A):
                    W = make_weights(cfg, a_i)
                    X = make_inputs(cfg, a_i)
                    st, lay = encodings.build(enc, W, m, sig, rng_device(cfg, a_i, enc, si, mi), cfg)
                    for ti, t in enumerate(t_grid):
                        for nm in (0, 1):
                            M, info = encodings.effective_matrix(st, lay, float(t), bool(nm), cfg)
                            Y = matmul(X, M, backend)
                            R_all[nm][ti, a_i] = checksum.residuals(Y, lay).astype(np.float32)
                            if enc == "c":
                                FLIP[ti, a_i, :, nm] = checksum.group_flip_flags(info["flips"], lay)
                        if enc == "c":
                            fm = encodings.sign_flip_matrices(st, lay, float(t), cfg)
                            PFLIP[ti, a_i] = [
                                fm["pcm_ref"].mean(),
                                fm["fixed_th_raw"].mean(),
                                fm["fixed_th_norm"].mean(),
                            ]

                for nm in (0, 1):
                    theta = metrics.calibrate_threshold(np.abs(R_all[nm][0]), cfg.fpr0)
                    thetas[(enc, si, mi, nm)] = theta
                    sigma_r_series = []
                    for ti, t in enumerate(t_grid):
                        r_t = R_all[nm][ti]
                        k_exc, n_tot = metrics.exceed_counts(r_t, theta)
                        rate, lo, hi = metrics.wilson_ci(k_exc, n_tot)
                        qs = metrics.signed_quantiles(r_t, cfg.quantile_levels)
                        sig_r = float(np.std(r_t, dtype=np.float64))
                        sigma_r_series.append(sig_r)
                        extra = {}
                        if enc == "c":
                            keep = ~FLIP[ti, :, :, nm]  # [A, K] groups with no flipped sign cell
                            r_cond = np.swapaxes(r_t, 1, 2)[keep]  # [n_keep, N]
                            kc, nc = metrics.exceed_counts(r_cond, theta)
                            extra = {
                                "p_flip_pcm_ref": float(PFLIP[ti, :, 0].mean()),
                                "p_flip_fixed_raw": float(PFLIP[ti, :, 1].mean()),
                                "p_flip_fixed_norm": float(PFLIP[ti, :, 2].mean()),
                                "n_cond": nc,
                                "n_exceed_cond": kc,
                                "rate_cond": kc / nc if nc else "",
                            }
                        rows.append(
                            row(
                                phase="A",
                                encoding=enc,
                                sigma_nu=sig,
                                m=m,
                                ratio_norm=nm,
                                t_rel=t,
                                n=n_tot,
                                n_exceed=k_exc,
                                rate=rate,
                                ci_lo=lo,
                                ci_hi=hi,
                                mean_r=float(np.mean(r_t, dtype=np.float64)),
                                sigma_r=sig_r,
                                q01=qs[0],
                                q05=qs[1],
                                q25=qs[2],
                                q50=qs[3],
                                q75=qs[4],
                                q95=qs[5],
                                q99=qs[6],
                                **extra,
                            )
                        )
                    slope, intercept, npts = metrics.late_slope(t_grid, sigma_r_series, cfg.slope_t_min)
                    slope_rows.append(
                        {
                            "encoding": enc,
                            "sigma_nu": sig,
                            "m": m,
                            "ratio_norm": nm,
                            "slope": slope,
                            "intercept": intercept,
                            "n_points": npts,
                            "sigma_r_last": sigma_r_series[-1],
                        }
                    )
                    npz_payload[f"r_s{si}_m{m}_norm{nm}"] = R_all[nm]
                if enc == "c":
                    npz_payload[f"flip_s{si}_m{m}"] = FLIP
                    npz_payload[f"pflip_s{si}_m{m}"] = PFLIP
                log.info(
                    "phase A: enc=%s sigma_nu=%g m=%d done (theta off/on = %.4g / %.4g)",
                    enc,
                    sig,
                    m,
                    thetas[(enc, si, mi, 0)],
                    thetas[(enc, si, mi, 1)],
                )

        npz_path = paths.run_dir / f"residuals_{enc}.npz"
        np.savez_compressed(npz_path, **npz_payload)
        log.info(
            "phase A: enc=%s finished in %.1f s; %s written (%.1f MB)",
            enc,
            time.perf_counter() - t_enc,
            npz_path.name,
            npz_path.stat().st_size / 2**20,
        )
    return thetas, rows, slope_rows


# ---------------------------------------------------------------------------
# Phase B
# ---------------------------------------------------------------------------


def _calibrate_slice(cfg: E1Config, enc: str, si: int, mi: int, backend: str):
    """t0-only threshold calibration for the Phase B slice (used when Phase A
    was not run in this invocation; identical arrays via the seed policy)."""
    m = cfg.m_grid[mi]
    sig = cfg.sigma_nu_grid[si]
    pools = {0: [], 1: []}
    for a_i in range(cfg.n_arrays):
        W = make_weights(cfg, a_i)
        X = make_inputs(cfg, a_i)
        st, lay = encodings.build(enc, W, m, sig, rng_device(cfg, a_i, enc, si, mi), cfg)
        for nm in (0, 1):
            M, _ = encodings.effective_matrix(st, lay, 1.0, bool(nm), cfg)
            pools[nm].append(np.abs(checksum.residuals(matmul(X, M, backend), lay)))
    return {nm: metrics.calibrate_threshold(np.concatenate([p.ravel() for p in pools[nm]]), cfg.fpr0) for nm in (0, 1)}


def run_phase_b(cfg: E1Config, paths, log, backend: str, thetas: dict):
    """Detection rate of the constant threshold under persistent single-cell faults."""
    t_grid = np.asarray(cfg.t_rel_grid, dtype=float)
    si = cfg.sigma_nu_grid.index(cfg.b_sigma_nu)
    mi = cfg.m_grid.index(cfg.b_m)
    conds = [("sa0", 0.0), ("sa1", 0.0)] + [("soft", d) for d in cfg.b_soft_d_grid]
    rows: list = []

    for enc in cfg.encodings:
        t_enc = time.perf_counter()
        th = {nm: thetas.get((enc, si, mi, nm)) for nm in (0, 1)}
        if th[0] is None or th[1] is None:
            log.info("phase B: enc=%s thresholds not in memory, running t0 calibration", enc)
            th = _calibrate_slice(cfg, enc, si, mi, backend)
        base = []
        for a_i in range(cfg.n_arrays):
            W = make_weights(cfg, a_i)
            X = make_inputs(cfg, a_i)[: cfg.b_n_mvm]
            st, lay = encodings.build(enc, W, cfg.b_m, cfg.b_sigma_nu, rng_device(cfg, a_i, enc, si, mi), cfg)
            base.append((st, lay, X))
        layout = base[0][1]

        for cond_id, (kind, d) in enumerate(conds):
            counts = {(ti, nm): [0, 0] for ti in range(len(t_grid)) for nm in (0, 1)}
            for trial in range(cfg.b_n_trials):
                a_i = trial % cfg.n_arrays
                st, lay, X = base[a_i]
                fr = rng_fault(cfg, enc, cond_id, trial)
                spec = faults.sample_fault(kind, d, lay, fr)
                stf, grp = faults.inject(st, lay, spec, cfg)
                for ti, t in enumerate(t_grid):
                    for nm in (0, 1):
                        M, _ = encodings.effective_matrix(stf, lay, float(t), bool(nm), cfg)
                        r_aff = checksum.residuals(matmul(X, M, backend), lay)[:, grp]
                        k_exc, n_tot = metrics.exceed_counts(r_aff, th[nm])
                        counts[(ti, nm)][0] += k_exc
                        counts[(ti, nm)][1] += n_tot
            for ti, t in enumerate(t_grid):
                for nm in (0, 1):
                    k_exc, n_tot = counts[(ti, nm)]
                    rate, lo, hi = metrics.wilson_ci(k_exc, n_tot)
                    rows.append(
                        row(
                            phase="B",
                            encoding=enc,
                            sigma_nu=cfg.b_sigma_nu,
                            m=cfg.b_m,
                            ratio_norm=nm,
                            t_rel=t,
                            fault_kind=kind,
                            d=d,
                            n=n_tot,
                            n_exceed=k_exc,
                            rate=rate,
                            ci_lo=lo,
                            ci_hi=hi,
                        )
                    )
            log.info("phase B: enc=%s cond=%s d=%.2f done", enc, kind, d)
        log.info("phase B: enc=%s finished in %.1f s", enc, time.perf_counter() - t_enc)
    return rows, {(enc,): None for enc in cfg.encodings}, conds


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def make_figures(cfg: E1Config, paths, log, rows_a: list):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m_show = cfg.b_m if cfg.b_m in cfg.m_grid else cfg.m_grid[0]
    encs = list(cfg.encodings)

    # Fig 1: FPR(t), panels ratio_norm (rows) x encoding (cols), lines sigma_nu
    fig, axes = plt.subplots(
        2, len(encs), figsize=(3.6 * len(encs), 6.2), squeeze=False, sharex=True, sharey=True
    )
    for ci, enc in enumerate(encs):
        for nm in (0, 1):
            ax = axes[nm][ci]
            for sig in cfg.sigma_nu_grid:
                pts = sorted(
                    (r["t_rel"], r["rate"], r["n"])
                    for r in rows_a
                    if r["phase"] == "A"
                    and r["encoding"] == enc
                    and r["sigma_nu"] == sig
                    and r["m"] == m_show
                    and r["ratio_norm"] == nm
                )
                if not pts:
                    continue
                t = [p[0] for p in pts]
                f = [max(p[1], 1.0 / (2.0 * p[2])) for p in pts]  # display floor for log axis
                ax.plot(t, f, marker="o", ms=3, lw=1.2, label=f"$\\sigma_\\nu$={sig}")
            ax.axhline(cfg.fpr0, ls="--", lw=0.8, color="gray")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"enc {enc}, ratio_norm={'on' if nm else 'off'}", fontsize=9)
            if nm == 1:
                ax.set_xlabel("$t/t_0$")
            if ci == 0:
                ax.set_ylabel("FPR")
    axes[0][0].legend(fontsize=7)
    fig.suptitle(f"E1 Phase A: constant-threshold FPR vs time (m={m_show})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        p = paths.figures / f"fig1_fpr_vs_t_{paths.runstamp}.{ext}"
        fig.savefig(p, dpi=200)
    plt.close(fig)
    log.info("figure written: fig1_fpr_vs_t_%s.pdf/.png (m=%d)", paths.runstamp, m_show)

    # P_flip figure (encoding c)
    if "c" in encs:
        rules = ["p_flip_pcm_ref", "p_flip_fixed_raw", "p_flip_fixed_norm"]
        titles = ["pcm_ref", "fixed_th (norm off)", "fixed_th (norm on)"]
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharex=True, sharey=True)
        for ri, (field, ttl) in enumerate(zip(rules, titles)):
            ax = axes[ri]
            for sig in cfg.sigma_nu_grid:
                pts = sorted(
                    (r["t_rel"], r[field])
                    for r in rows_a
                    if r["phase"] == "A"
                    and r["encoding"] == "c"
                    and r["sigma_nu"] == sig
                    and r["m"] == m_show
                    and r["ratio_norm"] == 0
                    and r[field] != ""
                )
                if not pts:
                    continue
                ax.plot(
                    [p[0] for p in pts],
                    [max(p[1], 1e-6) for p in pts],
                    marker="o",
                    ms=3,
                    lw=1.2,
                    label=f"$\\sigma_\\nu$={sig}",
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(ttl, fontsize=9)
            ax.set_xlabel("$t/t_0$")
            if ri == 0:
                ax.set_ylabel("$P_{flip}$")
                ax.legend(fontsize=7)
        fig.suptitle(f"E1: sign-cell flip probability, encoding c (m={m_show})", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        for ext in ("pdf", "png"):
            fig.savefig(paths.figures / f"pflip_vs_t_{paths.runstamp}.{ext}", dpi=200)
        plt.close(fig)
        log.info("figure written: pflip_vs_t_%s.pdf/.png", paths.runstamp)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def write_csv(path: Path, fieldnames: list, rows: list, log) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    log.info("written: %s (%d rows)", path.name, len(rows))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", nargs="+", choices=["A", "B"], default=["A", "B"])
    ap.add_argument("--enc", nargs="+", choices=["a", "b", "c"], default=["a", "b", "c"])
    ap.add_argument("--dense", action="store_true", help="use the 17-point half-decade time grid")
    ap.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    ap.add_argument("--out", default="results/E1", help="results root (default: results/E1)")
    ap.add_argument("--smoke", action="store_true", help="tiny run for pipeline verification")
    args = ap.parse_args()

    cfg = E1Config()
    cfg = replace(cfg, encodings=tuple(args.enc))
    if args.dense:
        cfg = replace(cfg, t_rel_grid=cfg.t_rel_grid_dense)
    if args.smoke:
        cfg = smoke_config(cfg)

    paths = io_utils.resolve_paths(args.out, exp="E1", suffix="_smoke" if args.smoke else "")
    log = io_utils.get_logger("E1", paths.logs / f"E1_{paths.runstamp}.log")
    t_start = time.perf_counter()
    log.info("E1 run %s starting (smoke=%s, phases=%s, enc=%s, backend=%s)",
             paths.runstamp, args.smoke, args.phase, args.enc, args.backend)
    log.info("run dir: %s", paths.run_dir)
    io_utils.save_json(
        io_utils.config_snapshot(cfg, vars(args), {"runstamp": paths.runstamp, "pcm_drift_version": pcm_drift.__version__}),
        paths.run_dir / "config.json",
    )

    invariant_selftest(cfg, log, args.backend)

    thetas: dict = {}
    rows: list = []
    slope_rows: list = []
    if "A" in args.phase:
        thetas, rows_a, slope_rows = run_phase_a(cfg, paths, log, args.backend)
        rows.extend(rows_a)
        make_figures(cfg, paths, log, rows_a)
        write_csv(
            paths.run_dir / "c2_slope_ranking.csv",
            ["encoding", "sigma_nu", "m", "ratio_norm", "slope", "intercept", "n_points", "sigma_r_last"],
            slope_rows,
            log,
        )
    if "B" in args.phase:
        rows_b, _, _ = run_phase_b(cfg, paths, log, args.backend, thetas)
        rows.extend(rows_b)

    write_csv(paths.run_dir / "summary.csv", SUMMARY_FIELDS, rows, log)
    theta_rows = [
        {"encoding": e, "sigma_nu": cfg.sigma_nu_grid[si], "m": cfg.m_grid[mi],
         "ratio_norm": nm, "theta": th, "fpr0": cfg.fpr0}
        for (e, si, mi, nm), th in sorted(thetas.items())
    ]
    if theta_rows:
        write_csv(
            paths.run_dir / "thresholds.csv",
            ["encoding", "sigma_nu", "m", "ratio_norm", "theta", "fpr0"],
            theta_rows,
            log,
        )
    log.info("E1 run %s complete in %.1f s; every number above traces to %s",
             paths.runstamp, time.perf_counter() - t_start, paths.run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
