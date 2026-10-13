#!/usr/bin/env python3
"""E8.1 (exploratory): detection latency of the online estimators after a fault appears.

Plan: docs/E8.1_Plan_2026-10-11.md (reading fixed before the run). E8 compares
the online variance-adaptive thresholds with the band at two end points, the
onset of a persistent fault and the steady state in which the estimator has
absorbed it. E8.1 follows the trajectory in between: it counts the inferences
from the appearance of a fault to the first alarm, for an abrupt fault and for
a deviation that grows linearly over T inferences. Version 2 metric (plan
amendment A-1): the excess alarm rate of the faulty stream over the healthy
stream of the same group under the same rule, per inference after onset, and
its cumulative sum (expected alarms the fault adds). The version 1 metric,
time to the first alarm (run 20261011_0444), counted false alarms as
detections and is not used.

Slice: encoding b, m = 16, c = 1, sigma_nu = 0.02, ratio readout; t in {1, 1e2};
delta in {0.3, 0.95}; the E4/E8 fault set (seed branch 8, same draw order).
The inference stream of each time point (2000 inferences) is traversed three
times circularly. The online estimators start from the steady state of the
healthy stream.

Run from the project root:

    python experiments/E8_1_transient.py --e8 results/E8/20261011_0440

Outputs per run: results/E8.1/<runstamp>/{config.json, transient.csv,
excess_traj.npz (per-inference excess alarm rate of every series)},
logs/E8.1_<runstamp>.log.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "src"))

import pcm_drift  # noqa: E402
from pcm_drift import io_utils, model  # noqa: E402

sys.path.insert(0, str(_HERE.parent))
from E1_config import E1Config  # noqa: E402
from E3_config import E3Config  # noqa: E402
from E8_config import E8Config  # noqa: E402
from E3_validity_domain_scan import ENC_ID, build_ensemble, physics_params  # noqa: E402
from E4_baseline_comparison import pools_with_yck  # noqa: E402
from E8_published_baselines import healthy_online  # noqa: E402

FIELDS = ["sigma_nu", "ratio_norm", "t_rel", "delta", "form", "ramp_T", "detector",
          "threshold", "margin_n0", "margin_W", "margin_4W", "margin_end",
          "excess_W", "excess_4W", "excess_total", "n_trials", "horizon", "W"]


def excess_trajectory(zf, zh, horizon, thr_static=None, k=None, v0=None, shift=None,
                      pooled_f=None, pooled_h=None):
    """Per-inference excess alarm rate of the faulty over the healthy stream.

    Both streams of every trial pass through the same rule: a static threshold
    (|z| > thr_static), or an online estimator started from the healthy steady
    state v0 and updated after each inference with z^2 (per group) or with the
    pooled per-array input (pooled_f / pooled_h). Returns the mean over trials
    of alarm_fault(n) - alarm_healthy(n) for n = 0 .. horizon-1. Under
    alarm-then-verify, every alarm triggers verification, so only the excess
    measures what the fault adds to the chance of being verified.
    """
    exc = np.empty(horizon)
    vf = None if v0 is None else v0.copy()
    vh = None if v0 is None else v0.copy()
    alpha = None if shift is None else 2.0 ** (-shift)
    for n in range(horizon):
        a_f, a_h = zf(n), zh(n)
        if thr_static is not None:
            exc[n] = (np.abs(a_f) > thr_static).mean() - (np.abs(a_h) > thr_static).mean()
            continue
        exc[n] = (np.abs(a_f) > k * np.sqrt(vf)).mean() - (np.abs(a_h) > k * np.sqrt(vh)).mean()
        xf = (a_f * a_f) if pooled_f is None else pooled_f(n, a_f)
        xh = (a_h * a_h) if pooled_h is None else pooled_h(n, a_h)
        vf = vf + alpha * (xf - vf)
        vh = vh + alpha * (xh - vh)
    return exc


def summarize_traj(exc, W):
    def win(a, b):
        return float(exc[a:b].mean())
    return {
        "margin_n0": win(0, 16),               # first 16 inferences after onset
        "margin_W": win(max(W - 8, 0), W + 8),
        "margin_4W": win(4 * W - 8, 4 * W + 8) if 4 * W + 8 <= exc.size else "",
        "margin_end": win(exc.size - 256, exc.size),
        "excess_W": float(exc[:W].sum()),
        "excess_4W": float(exc[:4 * W].sum()) if 4 * W <= exc.size else "",
        "excess_total": float(exc.sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/E8.1")
    ap.add_argument("--e8", default=None, help="E8 formal record (calibration cross-check)")
    args = ap.parse_args()

    cfg = E1Config()
    e3 = E3Config()
    e8 = E8Config()
    p = physics_params(cfg)
    sig, norm_on, m, c = 0.02, True, e8.m, e8.c
    lam = c / m
    t_list = [1.0, 100.0]
    shifts = list(e8.ewma_shifts)
    passes = e8.ewma_passes

    paths = io_utils.resolve_paths(args.out, exp="E8.1")
    log = io_utils.get_logger("E8.1", paths.logs / f"E8.1_{paths.runstamp}.log")
    t_start = time.perf_counter()
    log.info("E8.1 run %s starting", paths.runstamp)
    io_utils.save_json(io_utils.config_snapshot(e8, vars(args), {
        "runstamp": paths.runstamp, "slice": {"sigma_nu": sig, "ratio_norm": norm_on, "m": m, "c": c},
        "t_list": t_list, "pcm_drift_version": pcm_drift.__version__}), paths.run_dir / "config.json")

    si_g, mi_g, ci_g = e3.sigma_nu_grid.index(sig), e3.m_grid.index(m), e3.c_grid.index(c)
    arrays = build_ensemble("b", si_g, mi_g, ci_g, cfg, e3)
    t_grid = list(e3.t_rel_grid)
    r_off, _, s_arr = pools_with_yck(arrays, t_grid, cfg)
    A, N, K = len(arrays), arrays[0][2].shape[0], arrays[0][1].K
    horizon = passes * N

    z0 = s_arr[0][:, None, None] * r_off[0].astype(np.float64)
    theta0 = float(np.quantile(np.abs(z0), 1.0 - e8.fpr0))
    _, sd_t0 = model.r_moments("b", m, 0.0, sig, p, norm_on, lam_scale=c)
    k_of = {}
    for s_ in shifts:
        u_g, u_p, _, _ = healthy_online(z0, s_, passes)
        k_of[f"O_g{s_}"] = float(np.quantile(u_g, 1.0 - e8.fpr0))
        k_of[f"O_p{s_}"] = float(np.quantile(u_p, 1.0 - e8.fpr0))
    if args.e8:
        cal = json.loads((Path(args.e8) / "calibration.json").read_text(encoding="utf-8"))
        ref = cal[f"b/{sig}/1"]
        ok = all(ref[kk] == vv for kk, vv in k_of.items()) and ref["theta0"] == theta0
        log.info("calibration cross-check against E8 record: %s", "MATCH" if ok else "MISMATCH")
        if not ok:
            raise AssertionError("calibration differs from the E8 record")

    frng = np.random.default_rng(np.random.SeedSequence(
        [e3.seed_base, e8.fault_seed_branch, ENC_ID["b"], si_g, mi_g, ci_g, int(norm_on)]))
    rows = []
    traj = {}
    n_trials = e8.n_trials
    for ti, t in enumerate(t_grid):
        if t not in e8.probe_t:
            continue
        L = float(np.log(t))
        _, sd_t = model.r_moments("b", m, L, sig, p, norm_on, lam_scale=c)
        f_band = theta0 * (sd_t / sd_t0)
        z_t = s_arr[ti][:, None, None] * r_off[ti].astype(np.float64)
        st = {s_: healthy_online(z_t, s_, passes)[2:] for s_ in shifts}
        qh = (z_t * z_t).mean(axis=2)                                     # [A, N]
        for d in e8.probe_deltas:
            aa = frng.integers(0, A, n_trials)
            rows_i = frng.integers(0, cfg.R, n_trials)
            cols = frng.integers(0, cfg.C, n_trials)
            frng.integers(0, N, n_trials)                                 # nn (unused)
            br = frng.integers(0, 2, n_trials)
            sg = np.where(frng.integers(0, 2, n_trials) == 0, 1.0, -1.0)
            if t not in t_list:
                continue
            kk = cols // m
            rs = np.empty((n_trials, N)); xs = np.empty((n_trials, N))
            dm = np.empty(n_trials); sv = np.empty(n_trials)
            for a in range(A):
                sel = np.flatnonzero(aa == a)
                if sel.size == 0:
                    continue
                state, layout, X = arrays[a]
                j, i = cols[sel], rows_i[sel]
                pcol = np.where(br[sel] == 0, layout.pos_cols[j], layout.neg_cols[j])
                bsign = np.where(br[sel] == 0, 1.0, -1.0)
                g0, nu = state.g0[i, pcol], state.nu[i, pcol]
                dm[sel] = bsign * (np.maximum(g0 + sg[sel] * d, cfg.g0_floor) - g0) * np.exp(-nu * L)
                rs[sel] = r_off[ti, a][:, kk[sel]].astype(np.float64).T
                xs[sel] = X[:, i].T
                sv[sel] = s_arr[ti, a]
            shift_full = -lam * xs * dm[:, None]

            for form, ramps in (("abrupt", {s_: 0 for s_ in shifts}),
                                ("ramp", {s_: 4 * 2 ** s_ for s_ in shifts})):
                for s_ in shifts:
                    T = ramps[s_]
                    W = 2 ** s_

                    def zf(n, T=T):
                        g = 1.0 if T == 0 else min(1.0, (n + 1) / T)
                        nn_ = n % N
                        return sv * (rs[:, nn_] + g * shift_full[:, nn_])

                    def zh(n):
                        return sv * rs[:, n % N]

                    v_g, v_p = st[s_]
                    vg0 = v_g[aa, kk, 0]
                    vp0 = v_p[aa, 0]

                    def pooled_f(n, z, aa=aa):
                        nn_ = n % N
                        zh_ = sv * rs[:, nn_]
                        return qh[aa, nn_] + (z * z - zh_ * zh_) / K

                    def pooled_h(n, z, aa=aa):
                        return qh[aa, n % N]

                    res = [
                        ("C_band", f_band, excess_trajectory(zf, zh, horizon, thr_static=f_band)),
                        (f"O_g{s_}", k_of[f"O_g{s_}"],
                         excess_trajectory(zf, zh, horizon, k=k_of[f"O_g{s_}"], v0=vg0, shift=s_)),
                        (f"O_p{s_}", k_of[f"O_p{s_}"],
                         excess_trajectory(zf, zh, horizon, k=k_of[f"O_p{s_}"], v0=vp0, shift=s_,
                                           pooled_f=pooled_f, pooled_h=pooled_h)),
                    ]
                    for name, thr, exc in res:
                        if form == "abrupt" and name == "C_band" and s_ != shifts[0]:
                            continue  # the abrupt band trajectory does not depend on s
                        sm = summarize_traj(exc, W)
                        traj[f"t{t:g}_d{d}_{form}_T{T}_{name}"] = exc
                        rows.append({"sigma_nu": sig, "ratio_norm": int(norm_on), "t_rel": t,
                                     "delta": d, "form": form, "ramp_T": T, "detector": name,
                                     "threshold": thr, **sm, "n_trials": n_trials,
                                     "horizon": horizon, "W": W})
                        log.info("t=%.0e delta=%.2f %s T=%d W=%d %s: margin n0=%.4f W=%.4f end=%.4f;"
                                 " excess alarms W=%.2f total=%.2f",
                                 t, d, form, T, W, name, sm["margin_n0"], sm["margin_W"],
                                 sm["margin_end"], sm["excess_W"], sm["excess_total"])
    with open(paths.run_dir / "transient.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(paths.run_dir / "excess_traj.npz", **traj)
    log.info("written: transient.csv (%d rows), excess_traj.npz (%d series)", len(rows), len(traj))
    log.info("E8.1 run %s complete in %.1f s; every number above traces to %s",
             paths.runstamp, time.perf_counter() - t_start, paths.run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
