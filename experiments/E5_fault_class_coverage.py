#!/usr/bin/env python3
"""E5: fault-class coverage across encodings and the sign-flip phase boundary.

Implements docs/E5_Design_Spec_2026-08-30.md (v2, frozen). Five fault
conditions (sa0, sa1, soft_abs 0.3, soft_abs 0.95, sign_flip) on the three
encodings at the primary slice; encodings a/b are judged by the deployed
drift-aware band, encoding c by the per-t oracle empirical band (ruling P-A,
a fortiori structure). Separately, the lawful sign-flip rate and the group
contamination fraction are measured on flip ensembles and compared with the
closed-form p_flip model; the phase boundary t_c solves
contamination(t) = kappa * FPR0.

All faulted residuals use the exact shift identity r' = r - lam * x * dM(t);
a start-up audit rebuilds faulted arrays end to end and checks the identity to
1e-9 for every (encoding, class) rule used.

Run from the project root:

    python experiments/E5_fault_class_coverage.py            (full)
    python experiments/E5_fault_class_coverage.py --smoke    (pipeline check)
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
from pcm_drift.faults import FaultSpec, inject, sample_fault  # noqa: E402

sys.path.insert(0, str(_HERE.parent))
from E1_config import E1Config  # noqa: E402
from E3_config import E3Config  # noqa: E402
from E5_config import E5Config  # noqa: E402
from E3_validity_domain_scan import ENC_ID, build_ensemble, physics_params  # noqa: E402

P1_FIELDS = ["encoding", "fault", "delta", "t_rel", "detector", "band", "fpr",
             "p1_sim", "p1_model", "delta_p", "n90", "n_trials"]
FLIP_FIELDS = ["sigma_nu", "t_rel", "pflip_sim", "pflip_model", "pflip_sd_ref",
               "contam_sim", "contam_model", "n_cells", "n_groups"]


def class_dm(enc, kind, d, state, layout, cfg, p, i, j, br, sg, L):
    """Vectorized logical matrix-entry deviation dM(t) for trial index arrays
    (i rows, j logical data columns) under the frozen class semantics (spec
    section 2). Encodings a/b return the OFF-domain dM; encoding c returns the
    magnitude-path conductance deviation only (caller applies sigma_hat and
    the readout scale)."""
    if enc == "b":
        pcol_p = layout.pos_cols[j]
        pcol_n = layout.neg_cols[j]
        if kind == "sign_flip":
            return -2.0 * (state.g0[i, pcol_p] * np.exp(-state.nu[i, pcol_p] * L)
                           - state.g0[i, pcol_n] * np.exp(-state.nu[i, pcol_n] * L))
        pcol = np.where(br == 0, pcol_p, pcol_n)
        bsign = np.where(br == 0, 1.0, -1.0)
        g0 = state.g0[i, pcol]
        nu = state.nu[i, pcol]
    elif enc == "a":
        g0 = state.g0[i, j]
        nu = state.nu[i, j]
        if kind == "sign_flip":
            v = state.W[i, j]
            gbar = p.g_off + p.slope_a * v
            g0f = np.maximum((p.g_off - p.slope_a * v) * (g0 / gbar), cfg.g0_floor)
            return (g0f - g0) * np.exp(-nu * L)
        bsign = 1.0
    else:  # encoding c magnitude path (sa / soft_abs only here)
        g0 = state.g0[i, j]
        nu = state.nu[i, j]
        bsign = 1.0
    if kind == "soft_abs":
        g_new = np.maximum(g0 + sg * d, cfg.g0_floor)
        return bsign * (g_new - g0) * np.exp(-nu * L)
    if kind in ("sa0", "sa1"):
        g_st = cfg.g_min if kind == "sa0" else cfg.g_max
        return bsign * (g_st - g0 * np.exp(-nu * L))
    raise ValueError((enc, kind))


def shift_audit(cfg, p, e5, log):
    """End-to-end check of every dM rule used: rebuild the faulted array, run
    the real readout, compare against the analytic shift."""
    t_chk = 1e2
    L = float(np.log(t_chk))
    rng = np.random.default_rng(123)
    X = np.random.default_rng(7).random((e5.audit_samples, cfg.R))
    worst = 0.0
    for enc in e5.encodings:
        dev = np.random.default_rng(np.random.SeedSequence([e5.mc_seed, ENC_ID[enc]]))
        W = np.clip(dev.normal(0.0, cfg.w_sigma, (cfg.R, cfg.C)), -1.0, 1.0)
        state, layout = encodings.build(enc, W, e5.m, e5.sigma_nu, dev, cfg)
        norm = enc == "c"  # encoding c is judged in the ON domain
        M0, info0 = encodings.effective_matrix(state, layout, t_chk, norm, cfg)
        r0 = checksum.residuals(X @ M0, layout)
        for kind in ("soft_abs", "sign_flip"):
            d = 0.3 if kind == "soft_abs" else 0.0
            spec = sample_fault("soft_abs", d, layout, rng)
            i, j = spec.row, spec.col
            if kind == "soft_abs":
                spec = FaultSpec("soft_abs", d, i, j, spec.branch, spec.soft_sign)
                st2, _ = inject(state, layout, spec, cfg)
                br = np.array([0 if spec.branch != "neg" else 1])
                dm = class_dm(enc, kind, d, state, layout, cfg, p,
                              np.array([i]), np.array([j]), br,
                              np.array([float(spec.soft_sign)]), L)[0]
                if enc == "c":
                    dm = float(info0["sigma_hat"][i, j]) * info0["scale"] * dm
            else:
                st2 = encodings.CrossbarState(
                    g0=state.g0.copy(), nu=state.nu.copy(),
                    ref_ratio=state.ref_ratio, ref_sign=state.ref_sign,
                    sign_g0=None if state.sign_g0 is None else state.sign_g0.copy(),
                    sign_nu=state.sign_nu, sign_true=state.sign_true, W=state.W)
                if enc == "a":
                    v = state.W[i, j]
                    gbar = p.g_off + p.slope_a * v
                    st2.g0[i, j] = max((p.g_off - p.slope_a * v) * state.g0[i, j] / gbar,
                                       cfg.g0_floor)
                elif enc == "b":
                    pp, pn = layout.pos_cols[j], layout.neg_cols[j]
                    st2.g0[i, pp], st2.g0[i, pn] = state.g0[i, pn], state.g0[i, pp]
                    st2.nu[i, pp], st2.nu[i, pn] = state.nu[i, pn], state.nu[i, pp]
                else:
                    # force the comparator to the OPPOSITE side of sign_true
                    st2.sign_g0[i, j] = (cfg.g0_floor if state.sign_true[i, j] > 0
                                         else 10.0)
                if enc == "c":
                    dm = -2.0 * float(M0[i, j])
                else:
                    dm = class_dm(enc, kind, 0.0, state, layout, cfg, p,
                                  np.array([i]), np.array([j]), np.array([0]),
                                  np.array([1.0]), L)[0]
            M1, _ = encodings.effective_matrix(st2, layout, t_chk, norm, cfg)
            r1 = checksum.residuals(X @ M1, layout)
            shift = -layout.lam * X[:, i] * dm
            got = r1[:, j // layout.m] - r0[:, j // layout.m]
            err = float(np.abs(got - shift).max())
            log.info("audit enc=%s kind=%s: |diff| = %.2e", enc, kind, err)
            worst = max(worst, err)
    if worst > e5.audit_tol:
        raise AssertionError(f"shift-identity audit FAILED: worst {worst:.3e}")
    log.info("shift-identity audit PASS: worst |diff| = %.2e over all rules", worst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results/E5")
    args = ap.parse_args()

    cfg = E1Config()
    e3 = E3Config()
    e5 = E5Config()
    p = physics_params(cfg)
    t_grid = list(e3.t_rel_grid)
    t_dense = list(cfg.t_rel_grid_dense)
    n_trials = e5.n_trials
    n_flip = e5.n_arrays_flip
    flip_sigmas = list(e5.flip_sigma_grid)
    if args.smoke:
        t_grid = [1.0, 1e2, 1e8]
        t_dense = [1.0, 10.0, 1e2, 1e4, 1e8]
        e3 = dataclasses.replace(e3, n_arrays=4, n_mvm=300)
        n_trials, n_flip, flip_sigmas = 600, 60, [0.02]

    paths = io_utils.resolve_paths(args.out, exp="E5", suffix="_smoke" if args.smoke else "")
    log = io_utils.get_logger("E5", paths.logs / f"E5_{paths.runstamp}.log")
    t_start = time.perf_counter()
    log.info("E5 run %s starting (smoke=%s)", paths.runstamp, args.smoke)
    io_utils.save_json(
        io_utils.config_snapshot(e5, vars(args), {
            "runstamp": paths.runstamp, "physics": dataclasses.asdict(cfg),
            "e3_seed_coords": {"seed_base": e3.seed_base, "n_arrays": e3.n_arrays,
                               "n_mvm": e3.n_mvm},
            "pcm_drift_version": pcm_drift.__version__,
        }),
        paths.run_dir / "config.json",
    )

    checks = {
        "cell": model.mc_selftest(p, np.random.default_rng(e5.mc_seed), 400_000),
        "fault": model.mc_selftest_fault(p, np.random.default_rng(e5.mc_seed + 1), 400_000),
        "signflip": model.mc_selftest_signflip(p, np.random.default_rng(e5.mc_seed + 2), 400_000),
        "pflip": model.mc_selftest_pflip(p, np.random.default_rng(e5.mc_seed + 3)),
    }
    worst = max(v["worst_rel_err"] for v in checks.values())
    if worst > e5.mc_tol_rel:
        raise AssertionError(f"model selftests FAILED: {checks}")
    log.info("model selftests PASS: %s (tol %.2e)",
             {k: f"{v['worst_rel_err']:.2e}" for k, v in checks.items()}, e5.mc_tol_rel)
    shift_audit(cfg, p, e5, log)

    sig, m, cval = e5.sigma_nu, e5.m, e5.c
    lam = cval / m
    si_g = e3.sigma_nu_grid.index(sig)
    mi_g = e3.m_grid.index(m)
    ci_g = e3.c_grid.index(cval)
    rows_p1 = []
    dm_cache = {}

    for enc in e5.encodings:
        tc0 = time.perf_counter()
        arrays = build_ensemble(enc, si_g, mi_g, ci_g, cfg, e3)
        A = len(arrays)
        N = arrays[0][2].shape[0]
        K = arrays[0][1].K
        Ctot = arrays[0][1].C + K
        r_pool = np.empty((len(t_grid), A, N, K), dtype=np.float32)
        s_arr = np.empty((len(t_grid), A))
        M_on = sgh = None
        if enc == "c":
            M_on = np.empty((len(t_grid), A, cfg.R, Ctot), dtype=np.float32)
            sgh = np.empty((len(t_grid), A, cfg.R, Ctot), dtype=np.int8)
        for a, (state, layout, X) in enumerate(arrays):
            for ti, t in enumerate(t_grid):
                if enc == "c":
                    M, info = encodings.effective_matrix(state, layout, float(t), True, cfg)
                    r_pool[ti, a] = checksum.residuals(X @ M, layout).astype(np.float32)
                    M_on[ti, a] = M.astype(np.float32)
                    sgh[ti, a] = info["sigma_hat"].astype(np.int8)
                    s_arr[ti, a] = float(info["scale"])
                else:
                    M, _ = encodings.effective_matrix(state, layout, float(t), False, cfg)
                    r_pool[ti, a] = checksum.residuals(X @ M, layout).astype(np.float32)
                    s_arr[ti, a] = encodings._ratio_scale(state, float(t), True)

        if enc == "c":
            med = np.array([float(np.median(r_pool[ti])) for ti in range(len(t_grid))])
            band = [float(np.quantile(
                np.abs(r_pool[ti].astype(np.float64) - med[ti]), 1.0 - e5.fpr0))
                for ti in range(len(t_grid))]
            mu_hat = list(med)
            fpr_t = [float((np.abs(r_pool[ti].astype(np.float64) - med[ti]) > band[ti]).mean())
                     for ti in range(len(t_grid))]
            det = "oracle_quantile"
        else:
            base0 = np.abs(s_arr[0][:, None, None] * r_pool[0].astype(np.float64))
            theta0 = float(np.quantile(base0, 1.0 - e5.fpr0))
            _, sd0 = model.r_moments(enc, m, 0.0, sig, p, True, lam_scale=cval)
            band, mu_hat, fpr_t = [], [], []
            for ti, t in enumerate(t_grid):
                L = float(np.log(t))
                mu_t, sd_t = model.r_moments(enc, m, L, sig, p, True, lam_scale=cval)
                f = theta0 * sd_t / sd0
                mh = mu_t if enc == "a" else 0.0
                band.append(f)
                mu_hat.append(mh)
                stat = np.abs(s_arr[ti][:, None, None] * r_pool[ti].astype(np.float64) - mh)
                fpr_t.append(float((stat > f).mean()))
            det = "deployed_band"
        log.info("enc %s pools+calibration done (%.1f s), FPR range [%.4f, %.4f]",
                 enc, time.perf_counter() - tc0, min(fpr_t), max(fpr_t))

        frng = np.random.default_rng(np.random.SeedSequence(
            [e3.seed_base, e5.fault_seed_branch, ENC_ID[enc], si_g, mi_g, ci_g]))
        for kind, d in e5.classes:
            for ti, t in enumerate(t_grid):
                L = float(np.log(t))
                aa = frng.integers(0, A, n_trials)
                ii = frng.integers(0, cfg.R, n_trials)
                jj = frng.integers(0, cfg.C, n_trials)
                nn = frng.integers(0, N, n_trials)
                br = frng.integers(0, 2, n_trials)
                sgn = np.where(frng.integers(0, 2, n_trials) == 0, 1.0, -1.0)
                hits = 0
                for a in range(A):
                    sel = aa == a
                    if not sel.any():
                        continue
                    state, layout, X = arrays[a]
                    i, j, n_in = ii[sel], jj[sel], nn[sel]
                    k = j // layout.m
                    if enc == "c" and kind == "sign_flip":
                        dm_eff = -2.0 * M_on[ti, a, i, j].astype(np.float64)
                    elif enc == "c":
                        dmg = class_dm(enc, kind, d, state, layout, cfg, p,
                                       i, j, br[sel], sgn[sel], L)
                        dm_eff = sgh[ti, a, i, j].astype(np.float64) * s_arr[ti, a] * dmg
                    else:
                        dm_eff = class_dm(enc, kind, d, state, layout, cfg, p,
                                          i, j, br[sel], sgn[sel], L)
                    r1 = r_pool[ti, a, n_in, k].astype(np.float64) - lam * X[n_in, i] * dm_eff
                    if enc == "c":
                        hits += int((np.abs(r1 - mu_hat[ti]) > band[ti]).sum())
                    else:
                        hits += int((np.abs(s_arr[ti, a] * r1 - mu_hat[ti]) > band[ti]).sum())
                p1 = hits / n_trials
                if enc == "c":
                    p1m = ""
                else:
                    key = (enc, kind, round(float(d), 6), round(L, 9))
                    if key not in dm_cache:
                        dm_cache[key] = model.fault_dm_nodes(enc, kind, float(d), L, sig, p)
                    dmv, dmw = dm_cache[key]
                    p1m = model.predict_p1(enc, m, L, sig, p, True, band[ti], dmv, dmw,
                                           lam_scale=cval,
                                           mu_hat=(mu_hat[ti] if enc == "a" else 0.0))
                n90 = (np.log(0.1) / np.log(1.0 - p1)) if 0.0 < p1 < 1.0 else float("inf")
                rows_p1.append({
                    "encoding": enc, "fault": kind, "delta": d, "t_rel": t,
                    "detector": det, "band": float(band[ti]), "fpr": fpr_t[ti],
                    "p1_sim": p1, "p1_model": p1m, "delta_p": p1 - fpr_t[ti],
                    "n90": ("inf" if not np.isfinite(n90) else f"{n90:.6g}"),
                    "n_trials": n_trials,
                })
        log.info("enc %s fault classes done (%.1f s total)", enc, time.perf_counter() - tc0)

    rows_flip, tc_report = [], {}
    for si, sflip in enumerate(flip_sigmas):
        tcf = time.perf_counter()
        pf_sim = np.zeros(len(t_dense))
        ct_sim = np.zeros(len(t_dense))
        n_groups_tot = 0
        n_cells_tot = 0
        for a in range(n_flip):
            dev = np.random.default_rng(np.random.SeedSequence(
                [e3.seed_base, e5.flip_seed_branch, a, si]))
            W = np.clip(dev.normal(0.0, cfg.w_sigma, (cfg.R, cfg.C)), -1.0, 1.0)
            state, layout = encodings.build("c", W, e5.m, sflip, dev, cfg)
            Kf = layout.K
            if a == 0:
                n_cells_tot = n_flip * cfg.R * (layout.C + Kf)
                n_groups_tot = n_flip * Kf
            for ti, t in enumerate(t_dense):
                fl = encodings.sign_flip_matrices(state, layout, float(t), cfg)["pcm_ref"]
                pf_sim[ti] += fl.mean()
                for k in range(Kf):
                    cols = np.concatenate([np.arange(k * e5.m, (k + 1) * e5.m),
                                           [layout.C + k]])
                    ct_sim[ti] += bool(fl[:, cols].any())
        pf_sim /= n_flip
        ct_sim /= n_groups_tot
        pf_mod = [model.p_flip_model(float(np.log(t)), sflip, p, m=e5.m)
                  for t in t_dense]
        for ti, t in enumerate(t_dense):
            rows_flip.append({"sigma_nu": sflip, "t_rel": t,
                              "pflip_sim": pf_sim[ti], "pflip_model": pf_mod[ti]["p_flip"],
                              "pflip_sd_ref": pf_mod[ti]["p_flip_sd_ref"],
                              "contam_sim": ct_sim[ti],
                              "contam_model": pf_mod[ti]["contamination"],
                              "n_cells": n_cells_tot, "n_groups": n_groups_tot})
        tfine = np.logspace(0, 8, e5.n_tc_grid)
        cm = np.array([model.p_flip_model(float(np.log(t)), sflip, p, m=e5.m)["contamination"]
                       for t in tfine])
        thr = e5.kappa * e5.fpr0
        tc_model = float(np.exp(np.interp(thr, cm, np.log(tfine)))) if cm[-1] > thr else float("inf")
        cs = np.maximum.accumulate(ct_sim)
        tc_emp = float(np.exp(np.interp(thr, cs, np.log(t_dense)))) if cs[-1] > thr else float("inf")
        tc_report[sflip] = (tc_model, tc_emp)
        log.info("flip sigma=%.3g: t_c model %.4g, empirical %.4g (%.1f s)",
                 sflip, tc_model, tc_emp, time.perf_counter() - tcf)

    for name, rows, fields in (("p1_by_class.csv", rows_p1, P1_FIELDS),
                               ("flip_rate.csv", rows_flip, FLIP_FIELDS)):
        with open(paths.run_dir / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    io_utils.save_json({"kappa_fpr0": e5.kappa * e5.fpr0,
                        "t_c": {str(k): list(v) for k, v in tc_report.items()}},
                       paths.run_dir / "t_c.json")

    ab = [r for r in rows_p1 if r["encoding"] in ("a", "b")]
    derr = sorted(abs(r["p1_sim"] - r["p1_model"]) for r in ab)
    n_bad = sum(e > e5.tol_p1_abs for e in derr)
    log.info("verdict p1 (a/b, %d points): median %.4f / p90 %.4f / max %.4f / >tol %d",
             len(derr), derr[len(derr) // 2], derr[int(0.9 * len(derr))], derr[-1], n_bad)
    ok = bad = 0
    zmax = 0.0
    for r in rows_flip:
        if r["pflip_sim"] * r["n_cells"] >= e5.min_events:
            se = float(np.sqrt(r["pflip_sd_ref"] ** 2 / n_flip
                               + r["pflip_model"] / r["n_cells"]))
            z = abs(r["pflip_sim"] - r["pflip_model"]) / max(se, 1e-300)
            zmax = max(zmax, z)
            ok += z <= e5.z_gate
            bad += z > e5.z_gate
    log.info("verdict pflip (countable points, z test vs ref-ensemble SE): "
             "%d within z<=%.1f, %d outside, max z %.2f", ok, e5.z_gate, bad, zmax)
    log.info("written: p1_by_class.csv (%d rows), flip_rate.csv (%d rows), t_c.json",
             len(rows_p1), len(rows_flip))

    make_figures(rows_p1, rows_flip, tc_report, e5, paths, log)
    log.info("E5 run %s complete in %.1f s; every number above traces to %s",
             paths.runstamp, time.perf_counter() - t_start, paths.run_dir)
    return 0


def make_figures(rows_p1, rows_flip, tc_report, e5, paths, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cls_style = {("sa0", 0.0): ("tab:purple", "sa0"), ("sa1", 0.0): ("tab:red", "sa1"),
                 ("soft_abs", 0.3): ("tab:orange", "soft $\\delta$=0.3"),
                 ("soft_abs", 0.95): ("tab:brown", "soft $\\delta$=0.95"),
                 ("sign_flip", 0.0): ("tab:green", "sign flip")}
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3), sharey=True)
    for ax, enc in zip(axes, e5.encodings):
        for (kind, d), (col, lab) in cls_style.items():
            pts = sorted((r["t_rel"], max(r["p1_sim"], 5e-4)) for r in rows_p1
                         if r["encoding"] == enc and r["fault"] == kind and r["delta"] == d)
            ax.plot([q[0] for q in pts], [q[1] for q in pts], "o-", color=col, ms=3.5,
                    lw=1.1, label=lab)
        fp = sorted((r["t_rel"], max(r["fpr"], 5e-4)) for r in rows_p1
                    if r["encoding"] == enc and r["fault"] == "sa0")
        ax.plot([q[0] for q in fp], [q[1] for q in fp], "k--", lw=0.9, label="FPR")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$t/t_0$")
        band_name = "oracle band" if enc == "c" else "deployed band"
        ax.set_title(f"encoding {enc} ({band_name})", fontsize=9)
    axes[0].set_ylabel("$p_1$ per MVM")
    axes[0].legend(fontsize=6.5)
    fig.suptitle("E5: fault-class coverage ($\\sigma_\\nu$=0.02, m=16, norm on)", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(paths.figures / f"fig_e5_class_coverage_{paths.runstamp}.{ext}", dpi=200)
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    sig_set = sorted({r["sigma_nu"] for r in rows_flip})
    prim = e5.sigma_nu if e5.sigma_nu in sig_set else sig_set[0]
    sub = sorted((r["t_rel"], r) for r in rows_flip if r["sigma_nu"] == prim)
    ts = [q[0] for q in sub]
    val = lambda key: [max(q[1][key], 1e-8) for q in sub]
    ax1.plot(ts, val("pflip_model"), "-", color="tab:blue", lw=1.3, label="$p_{flip}$ model")
    ax1.plot(ts, val("pflip_sim"), "o", color="tab:blue", ms=4)
    ax1.plot(ts, val("contam_model"), "-", color="tab:red", lw=1.3, label="contamination model")
    ax1.plot(ts, val("contam_sim"), "s", color="tab:red", ms=4)
    ax1.axhline(e5.kappa * e5.fpr0, color="k", ls=":", lw=1.0, label="$\\kappa\\,\\mathrm{FPR}_0$")
    if prim in tc_report and np.isfinite(tc_report[prim][0]):
        ax1.axvline(tc_report[prim][0], color="gray", lw=1.0)
        ax1.text(tc_report[prim][0] * 1.25, 2e-8, "$t_c$", fontsize=9)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("$t/t_0$")
    ax1.set_title(f"$\\sigma_\\nu$={prim}", fontsize=9)
    ax1.legend(fontsize=7)
    for s in sig_set:
        sub = sorted((r["t_rel"], max(r["contam_sim"], 1e-8), max(r["contam_model"], 1e-8))
                     for r in rows_flip if r["sigma_nu"] == s)
        ax2.plot([q[0] for q in sub], [q[2] for q in sub], "-", lw=1.1, label=f"$\\sigma_\\nu$={s}")
        ax2.plot([q[0] for q in sub], [q[1] for q in sub], ".", ms=4,
                 color=ax2.lines[-1].get_color())
        if s in tc_report and np.isfinite(tc_report[s][0]):
            ax2.axvline(tc_report[s][0], color=ax2.lines[-1].get_color(), lw=0.7, ls="--")
    ax2.axhline(e5.kappa * e5.fpr0, color="k", ls=":", lw=1.0)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("$t/t_0$")
    ax2.set_title("group contamination and $t_c$ per $\\sigma_\\nu$", fontsize=9)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(paths.figures / f"fig_e5_flip_confound_{paths.runstamp}.{ext}", dpi=200)
    plt.close(fig)
    log.info("figures written to %s", paths.figures)


if __name__ == "__main__":
    sys.exit(main())
