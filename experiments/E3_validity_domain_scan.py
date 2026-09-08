#!/usr/bin/env python3
"""E3: validity-domain boundary scan under the drift-aware band.

Implements docs/E3_Design_Spec_2026-08-29.md (frozen revision). For each
configuration (encoding, sigma_nu, m, c, readout) the script:

  1. builds a fresh ensemble (independent seed branches from E1/E2),
  2. calibrates theta0 at t0 for the per-MVM alarm budget FPR0,
  3. applies the deployment band f(t) = theta0 * rho_model(t) and backtests
     the healthy false-alarm rate over the full time grid,
  4. computes the model prediction of the per-MVM hit probability p1(delta, t)
     for a persistent single data-cell fault (absolute deviation delta) and
     solves the minimum detectable amplitude d_min per latency budget N*,
  5. measures p1 by exact-shift Monte Carlo on the simulated ensemble
     (r' = r - lam * x * dM, exact by MVM linearity) on an adaptive delta set
     centered on the model prediction, and compares d_min against the model
     under the pre-registered tolerances.

Run from the project root:

    python experiments/E3_validity_domain_scan.py            (full)
    python experiments/E3_validity_domain_scan.py --smoke    (pipeline check)

Outputs per run: results/E3/<runstamp>/{config.json, dmin_grid.csv,
p1_curves.csv, fpr_backtest.csv}, figures/E3/*_<runstamp>.{pdf,png},
logs/E3_<runstamp>.log. Smoke runs carry the _smoke runstamp suffix.
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
from pcm_drift.faults import FaultSpec, faulted_cell  # noqa: E402

sys.path.insert(0, str(_HERE.parent))
from E1_config import E1Config  # noqa: E402
from E3_config import E3Config  # noqa: E402

ENC_ID = {"a": 0, "b": 1, "c": 2}

DMIN_FIELDS = [
    "encoding", "sigma_nu", "m", "c", "ratio_norm", "t_rel", "theta0", "rho", "f_band",
    "clip_frac", "dmin_model_5", "dmin_model_20", "dmin_model_100",
    "dmin_sim_5", "dmin_sim_20", "dmin_sim_100", "rel_err_20", "p1_at_ceiling_sim", "flag",
]


def p1_star(n_star: int) -> float:
    return 1.0 - 0.1 ** (1.0 / n_star)


def physics_params(cfg: E1Config) -> model.ModelParams:
    d = dataclasses.asdict(cfg)
    d["nu_clip"] = tuple(d["nu_clip"])
    return model.ModelParams.from_run_config({"config": d})


def build_ensemble(enc, si, mi, ci, cfg, e3):
    """Build A arrays + input pools; return per-array state/layout/X."""
    sig = e3.sigma_nu_grid[si]
    m = e3.m_grid[mi]
    c = e3.c_grid[ci]
    arrays = []
    for a in range(e3.n_arrays):
        dev = np.random.default_rng(np.random.SeedSequence(
            [e3.seed_base, 5, a, ENC_ID[enc], si, mi, ci]))
        W = np.clip(dev.normal(0.0, cfg.w_sigma, (cfg.R, cfg.C)), -1.0, 1.0)
        state, layout = encodings.build(enc, W, m, sig, dev, cfg, lam_scale=c)
        xin = np.random.default_rng(np.random.SeedSequence([e3.seed_base, 6, a]))
        X = xin.random((e3.n_mvm, cfg.R))
        arrays.append((state, layout, X))
    return arrays


def healthy_pools(arrays, t_grid, cfg):
    """Residual pools per t (OFF-side) and per-(t, array) ratio scales."""
    A = len(arrays)
    K = arrays[0][1].K
    N = arrays[0][2].shape[0]
    r_off = np.empty((len(t_grid), A, N, K), dtype=np.float32)
    s_arr = np.empty((len(t_grid), A), dtype=np.float64)
    for a, (state, layout, X) in enumerate(arrays):
        for ti, t in enumerate(t_grid):
            M, info = encodings.effective_matrix(state, layout, float(t), False, cfg)
            Y = X @ M
            r_off[ti, a] = checksum.residuals(Y, layout).astype(np.float32)
            s_arr[ti, a] = encodings._ratio_scale(state, float(t), True)
    return r_off, s_arr


def sim_p1(arrays, r_off, s_arr, ti, t, deltas, kind, f_band, norm_on, mu_hat, lam, cfg, e3, frng):
    """Exact-shift Monte Carlo estimate of p1 per delta."""
    A = len(arrays)
    N = arrays[0][2].shape[0]
    L = float(np.log(t))
    out = []
    for d in deltas:
        n_tr = e3.n_trials
        aa = frng.integers(0, A, n_tr)
        rows = frng.integers(0, cfg.R, n_tr)
        cols = frng.integers(0, cfg.C, n_tr)
        nn = frng.integers(0, N, n_tr)
        br = frng.integers(0, 2, n_tr)
        sg = np.where(frng.integers(0, 2, n_tr) == 0, 1.0, -1.0)
        hits = 0
        for a in range(A):
            sel = aa == a
            if not sel.any():
                continue
            state, layout, X = arrays[a]
            i = rows[sel]; j = cols[sel]; n_in = nn[sel]
            if layout.encoding == "b":
                pcol = np.where(br[sel] == 0, layout.pos_cols[j], layout.neg_cols[j])
                bsign = np.where(br[sel] == 0, 1.0, -1.0)
            else:
                pcol = j
                bsign = np.ones(sel.sum())
            g0 = state.g0[i, pcol]
            nu = state.nu[i, pcol]
            F = np.exp(-nu * L)
            if kind == "soft_abs":
                g_new = np.maximum(g0 + sg[sel] * d, cfg.g0_floor)
                dm = bsign * (g_new - g0) * F
            elif kind in ("sa0", "sa1"):
                g_st = cfg.g_min if kind == "sa0" else cfg.g_max
                dm = bsign * (g_st - g0 * F)
            else:
                raise ValueError(kind)
            shift = -lam * X[n_in, i] * dm
            r1 = r_off[ti, a, n_in, j // layout.m].astype(np.float64) + shift
            if norm_on:
                stat = np.abs(s_arr[ti, a] * r1 - mu_hat)
            else:
                stat = np.abs(r1 - mu_hat)
            hits += int((stat > f_band).sum())
        out.append(hits / e3.n_trials)
    return out


def dmin_from_curve(deltas, p1s, target, ceiling):
    d = np.asarray(deltas, dtype=float)
    p = np.maximum.accumulate(np.asarray(p1s, dtype=float))  # enforce monotone
    if p[-1] < target:
        return float("inf")
    if p[0] >= target:
        return d[0]  # at or below grid start
    return float(np.exp(np.interp(target, p, np.log(d))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results/E3")
    args = ap.parse_args()

    cfg = E1Config()
    e3 = E3Config()
    p = physics_params(cfg)
    t_grid = list(e3.t_rel_grid)
    sig_grid, m_grid, c_grid = list(e3.sigma_nu_grid), list(e3.m_grid), list(e3.c_grid)
    A, N, n_trials = e3.n_arrays, e3.n_mvm, e3.n_trials
    if args.smoke:
        sig_grid, m_grid, c_grid = [0.02], [16], [1, 3]
        t_grid = [1.0, 1e2, 1e4, 1e8]
        e3 = dataclasses.replace(e3, n_arrays=4, n_mvm=300, n_trials=600)

    paths = io_utils.resolve_paths(args.out, exp="E3", suffix="_smoke" if args.smoke else "")
    log = io_utils.get_logger("E3", paths.logs / f"E3_{paths.runstamp}.log")
    t_start = time.perf_counter()
    log.info("E3 run %s starting (smoke=%s)", paths.runstamp, args.smoke)
    io_utils.save_json(
        io_utils.config_snapshot(e3, vars(args), {
            "runstamp": paths.runstamp, "physics": dataclasses.asdict(cfg),
            "pcm_drift_version": pcm_drift.__version__,
            "p1_star": {n: p1_star(n) for n in e3.n_star_grid},
        }),
        paths.run_dir / "config.json",
    )

    st1 = model.mc_selftest(p, np.random.default_rng(e3.mc_seed), 400_000)
    st2 = model.mc_selftest_fault(p, np.random.default_rng(e3.mc_seed + 1), 400_000)
    if max(st1["worst_rel_err"], st2["worst_rel_err"]) > e3.mc_tol_rel:
        raise AssertionError(f"model selftests FAILED: {st1} {st2}")
    log.info("model selftests PASS: cell %.2e, fault %.2e (tol %.2e)",
             st1["worst_rel_err"], st2["worst_rel_err"], e3.mc_tol_rel)

    # configuration list: (enc, si, mi, ci, norm)
    confs = []
    for si in range(len(sig_grid)):
        for mi in range(len(m_grid)):
            for ci in range(len(c_grid)):
                confs.append(("b", si, mi, ci, True))
    if not args.smoke:
        osl = e3.off_slice
        confs.append(("b", sig_grid.index(osl[0]), m_grid.index(osl[1]), c_grid.index(osl[2]), False))
        for si in range(len(sig_grid)):
            confs.append(("a", si, m_grid.index(16), c_grid.index(1), True))
    else:
        confs.append(("a", 0, 0, 0, True))

    delta_grid_model = np.exp(np.linspace(
        np.log(e3.delta_model_grid_lo), np.log(e3.delta_ceiling), e3.n_delta_model))
    rows_dmin, rows_p1, rows_fpr, n_flag = [], [], [], 0
    dm_cache = {}

    def dm_nodes(enc_, kind_, d_, L_, sig_):
        key = (enc_, kind_, round(float(d_), 9), round(float(L_), 9), sig_)
        if key not in dm_cache:
            dm_cache[key] = model.fault_dm_nodes(enc_, kind_, float(d_), L_, sig_, p)
        return dm_cache[key]
    stars = {n: p1_star(n) for n in e3.n_star_grid}

    for enc, si, mi, ci, norm_on in confs:
        sig, m, c = sig_grid[si], m_grid[mi], c_grid[ci]
        lam = c / m
        tc = time.perf_counter()
        # map smoke-reduced indices onto full-grid seed coordinates for clarity
        si_g = e3.sigma_nu_grid.index(sig); mi_g = e3.m_grid.index(m); ci_g = e3.c_grid.index(c)
        arrays = build_ensemble(enc, si_g, mi_g, ci_g, cfg, dataclasses.replace(
            e3, n_arrays=e3.n_arrays, n_mvm=e3.n_mvm))
        r_off, s_arr = healthy_pools(arrays, t_grid, cfg)
        frng = np.random.default_rng(np.random.SeedSequence(
            [e3.seed_base, 7, ENC_ID[enc], si_g, mi_g, ci_g, int(norm_on)]))

        # t0 calibration
        if norm_on:
            base0 = np.abs(s_arr[0][:, None, None] * r_off[0].astype(np.float64))
        else:
            base0 = np.abs(r_off[0].astype(np.float64))
        theta0 = float(np.quantile(base0, 1.0 - e3.fpr0))
        _, sd_t0 = model.r_moments(enc, m, 0.0, sig, p, norm_on, lam_scale=c)

        for ti, t in enumerate(t_grid):
            L = float(np.log(t))
            mu_t, sd_t = model.r_moments(enc, m, L, sig, p, norm_on, lam_scale=c)
            rho = sd_t / sd_t0
            f_band = theta0 * rho
            mu_hat = mu_t if enc == "a" else 0.0
            dmm = model.delta_moments(enc, m, L, sig, p, lam_scale=c)
            # healthy FPR backtest
            if norm_on:
                stat = np.abs(s_arr[ti][:, None, None] * r_off[ti].astype(np.float64) - mu_hat)
            else:
                stat = np.abs(r_off[ti].astype(np.float64) - mu_hat)
            fpr = float((stat > f_band).mean())
            rows_fpr.append({"encoding": enc, "sigma_nu": sig, "m": m, "c": c,
                             "ratio_norm": int(norm_on), "t_rel": t, "theta0": theta0,
                             "rho": rho, "f_band": f_band, "fpr": fpr})

            # model p1 curve and d_min
            p1_model = []
            for d in delta_grid_model:
                dm, dw = dm_nodes(enc, "soft_abs", d, L, sig)
                p1_model.append(model.predict_p1(
                    enc, m, L, sig, p, norm_on, f_band, dm, dw, lam_scale=c,
                    mu_hat=(mu_hat if enc == "a" else 0.0)))
            dmin_mod = {n: dmin_from_curve(delta_grid_model, p1_model, stars[n], e3.delta_ceiling)
                        for n in e3.n_star_grid}

            # adaptive sim delta set
            base = dmin_mod[e3.n_star_primary]
            if np.isfinite(base):
                dset = sorted(set(
                    [min(max(f * base, e3.delta_model_grid_lo), e3.delta_ceiling)
                     for f in e3.sim_delta_factors] + [e3.delta_ceiling]))
            else:
                dset = list(e3.closed_probe)
            p1_sim = sim_p1(arrays, r_off, s_arr, ti, t, dset, "soft_abs",
                            f_band, norm_on, mu_hat, lam, cfg, e3, frng)
            p1_mod_at = []
            for d in dset:
                dm, dw = dm_nodes(enc, "soft_abs", d, L, sig)
                p1_mod_at.append(model.predict_p1(
                    enc, m, L, sig, p, norm_on, f_band, dm, dw, lam_scale=c,
                    mu_hat=(mu_hat if enc == "a" else 0.0)))
            for d, ps, pm in zip(dset, p1_sim, p1_mod_at):
                rows_p1.append({"encoding": enc, "sigma_nu": sig, "m": m, "c": c,
                                "ratio_norm": int(norm_on), "t_rel": t, "delta": d,
                                "p1_sim": ps, "p1_model": pm, "n_trials": e3.n_trials})
            dmin_sim = {n: dmin_from_curve(dset, p1_sim, stars[n], e3.delta_ceiling)
                        for n in e3.n_star_grid}

            n20m, n20s = dmin_mod[e3.n_star_primary], dmin_sim[e3.n_star_primary]
            rel = (n20s / n20m - 1.0) if np.isfinite(n20m) and np.isfinite(n20s) else float("nan")
            flags = []
            if np.isfinite(rel) and abs(rel) > e3.tol_dmin_rel:
                flags.append("DMIN")
            if any(abs(ps - pm) > e3.tol_p1_abs for ps, pm in zip(p1_sim, p1_mod_at)):
                flags.append("P1")
            gate = (e3.fpr_gate_hi_sigma_on if (sig == 0.04 and norm_on) else e3.fpr_gate)
            if not (gate[0] <= fpr <= gate[1]):
                flags.append("FPR")
            if (np.isfinite(n20m) != np.isfinite(n20s)):
                flags.append("CLOSURE")
            n_flag += bool(flags)
            fmt = lambda v: (f"{v:.6g}" if np.isfinite(v) else "closed")
            rows_dmin.append({
                "encoding": enc, "sigma_nu": sig, "m": m, "c": c, "ratio_norm": int(norm_on),
                "t_rel": t, "theta0": theta0, "rho": rho, "f_band": f_band,
                "clip_frac": dmm.get("clip_frac", 0.0),
                "dmin_model_5": fmt(dmin_mod[5]), "dmin_model_20": fmt(dmin_mod[20]),
                "dmin_model_100": fmt(dmin_mod[100]),
                "dmin_sim_5": fmt(dmin_sim[5]), "dmin_sim_20": fmt(dmin_sim[20]),
                "dmin_sim_100": fmt(dmin_sim[100]),
                "rel_err_20": rel, "p1_at_ceiling_sim": p1_sim[-1] if dset[-1] >= e3.delta_ceiling else "",
                "flag": "+".join(flags),
            })
        log.info("config enc=%s sigma=%.3g m=%d c=%d norm=%s done in %.1f s",
                 enc, sig, m, c, "on" if norm_on else "off", time.perf_counter() - tc)

        # sa reference curves on the designated slice
        if (not args.smoke and enc == "b" and norm_on
                and (sig, m, c) == e3.sa_slice):
            for kind in ("sa0", "sa1"):
                for ti, t in enumerate(t_grid):
                    L = float(np.log(t))
                    mu_t, sd_t = model.r_moments(enc, m, L, sig, p, True, lam_scale=c)
                    f_band = theta0 * (sd_t / sd_t0)
                    dm, dw = dm_nodes(enc, kind, 0.0, L, sig)
                    pm = model.predict_p1(enc, m, L, sig, p, True, f_band, dm, dw, lam_scale=c)
                    ps = sim_p1(arrays, r_off, s_arr, ti, t, [0.0], kind,
                                f_band, True, 0.0, lam, cfg, e3, frng)[0]
                    rows_p1.append({"encoding": enc, "sigma_nu": sig, "m": m, "c": c,
                                    "ratio_norm": 1, "t_rel": t, "delta": kind,
                                    "p1_sim": ps, "p1_model": pm, "n_trials": e3.n_trials})
            log.info("sa reference curves done")

    for name, rows, fields in (
        ("dmin_grid.csv", rows_dmin, DMIN_FIELDS),
        ("p1_curves.csv", rows_p1, ["encoding", "sigma_nu", "m", "c", "ratio_norm",
                                    "t_rel", "delta", "p1_sim", "p1_model", "n_trials"]),
        ("fpr_backtest.csv", rows_fpr, ["encoding", "sigma_nu", "m", "c", "ratio_norm",
                                        "t_rel", "theta0", "rho", "f_band", "fpr"]),
    ):
        with open(paths.run_dir / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    log.info("written: dmin_grid.csv (%d rows, %d flagged), p1_curves.csv (%d), fpr_backtest.csv (%d)",
             len(rows_dmin), n_flag, len(rows_p1), len(rows_fpr))

    fin = [r for r in rows_dmin if isinstance(r["rel_err_20"], float) and np.isfinite(r["rel_err_20"])]
    if fin:
        er = np.abs([r["rel_err_20"] for r in fin])
        log.info("verdict d_min(N*=20): finite cells %d/%d; |rel err| median %.3f / p90 %.3f / max %.3f",
                 len(fin), len(rows_dmin), np.median(er), np.percentile(er, 90), er.max())
    fprs = np.array([r["fpr"] for r in rows_fpr])
    log.info("verdict FPR backtest: min %.4f / median %.4f / max %.4f", fprs.min(), np.median(fprs), fprs.max())

    make_figures(rows_dmin, rows_p1, t_grid, sig_grid, m_grid, c_grid, e3, paths, log, args.smoke)
    log.info("E3 run %s complete in %.1f s; every number above traces to %s",
             paths.runstamp, time.perf_counter() - t_start, paths.run_dir)
    return 0


def make_figures(rows_dmin, rows_p1, t_grid, sig_grid, m_grid, c_grid, e3, paths, log, smoke):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def cell(enc, sig, m, c, nm, t, key):
        for r in rows_dmin:
            if (r["encoding"], r["sigma_nu"], r["m"], r["c"], r["ratio_norm"], r["t_rel"]) == (enc, sig, m, c, nm, t):
                return r[key]
        return None

    # Fig 2 candidate: model d_min heatmaps over (t, sigma) per (m, c), sim agreement overlay
    fig, axes = plt.subplots(len(m_grid), len(c_grid),
                             figsize=(3.0 * len(c_grid), 2.6 * len(m_grid)),
                             squeeze=False, sharex=True, sharey=True)
    for mi, m in enumerate(m_grid):
        for ci, c in enumerate(c_grid):
            Z = np.full((len(sig_grid), len(t_grid)), np.nan)
            for siy, sig in enumerate(sig_grid):
                for tix, t in enumerate(t_grid):
                    v = cell("b", sig, m, c, 1, t, "dmin_model_20")
                    Z[siy, tix] = np.log10(float(v)) if v not in (None, "closed") else np.nan
            ax = axes[mi][ci]
            im = ax.imshow(Z, aspect="auto", origin="lower", vmin=-2.3, vmax=0.0, cmap="viridis")
            for siy, sig in enumerate(sig_grid):
                for tix, t in enumerate(t_grid):
                    fl = cell("b", sig, m, c, 1, t, "flag") or ""
                    v = cell("b", sig, m, c, 1, t, "dmin_model_20")
                    if v == "closed":
                        ax.plot(tix, siy, "x", color="crimson", ms=5)
                    elif fl:
                        ax.plot(tix, siy, ".", color="orange", ms=4)
            ax.set_title(f"m={m}, c={c}", fontsize=8)
            ax.set_xticks(range(0, len(t_grid), 2))
            ax.set_xticklabels([f"1e{int(np.log10(t))}" for t in t_grid[::2]], fontsize=6)
            ax.set_yticks(range(len(sig_grid)))
            ax.set_yticklabels([str(s) for s in sig_grid], fontsize=6)
    fig.colorbar(im, ax=axes, label=r"$\log_{10} d_{\min}$ (N*=20)", shrink=0.8)
    fig.suptitle("E3: minimum detectable deviation under the drift-aware band\n"
                 "(x = closed at ceiling; orange dot = flagged vs sim)", fontsize=9)
    for ext in ("pdf", "png"):
        fig.savefig(paths.figures / f"fig_e3_validity_domain_{paths.runstamp}.{ext}", dpi=200)
    plt.close(fig)

    # d_min vs t per c (m=16, sigma=0.02): model lines, sim dots
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    for c in c_grid:
        xs, ym, ysim = [], [], []
        for t in t_grid:
            vm = cell("b", 0.02, 16, c, 1, t, "dmin_model_20")
            vs = cell("b", 0.02, 16, c, 1, t, "dmin_sim_20")
            if vm not in (None, "closed"):
                xs.append(t); ym.append(float(vm))
                ysim.append(float(vs) if vs not in (None, "closed") else np.nan)
        if xs:
            ax.plot(xs, ym, color=colors.get(c, "k"), lw=1.4, label=f"model c={c}")
            ax.plot(xs, ysim, "o", color=colors.get(c, "k"), ms=4)
    ax.axhline(e3.delta_ceiling, color="gray", ls="--", lw=0.8)
    if not ax.get_legend_handles_labels()[0]:
        ax.text(0.5, 0.5, "all cells closed on this slice", transform=ax.transAxes,
                ha="center", fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$t/t_0$"); ax.set_ylabel(r"$d_{\min}$  (N*=20)")
    ax.set_title("encoding b, m=16, $\\sigma_\\nu$=0.02, norm on", fontsize=9)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(paths.figures / f"fig_e3_dmin_vs_lambda_{paths.runstamp}.{ext}", dpi=200)
    plt.close(fig)

    # sa reference + soft contrast
    sa = [r for r in rows_p1 if r["delta"] in ("sa0", "sa1")]
    if sa:
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        for kind, col in (("sa0", "tab:purple"), ("sa1", "tab:red")):
            pts = sorted([(r["t_rel"], r["p1_sim"], r["p1_model"]) for r in sa if r["delta"] == kind])
            ax.plot([q[0] for q in pts], [max(q[2], 1e-4) for q in pts], color=col, lw=1.3, label=f"{kind} model")
            ax.plot([q[0] for q in pts], [max(q[1], 1e-4) for q in pts], "o", color=col, ms=4)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("$t/t_0$"); ax.set_ylabel("$p_1$")
        ax.set_title("stuck-fault reference, encoding b (m=16, $\\sigma_\\nu$=0.02, c=1, norm on)", fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(paths.figures / f"fig_e3_sa_reference_{paths.runstamp}.{ext}", dpi=200)
        plt.close(fig)
    log.info("figures written to %s", paths.figures)


if __name__ == "__main__":
    sys.exit(main())
