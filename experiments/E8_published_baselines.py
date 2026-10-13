#!/usr/bin/env python3
"""E8: decision rules of published approaches on seed-identical E3 ensembles.

Implements docs/E8_Design_Spec_2026-10-11.md (frozen before the formal run).
E4 compared the band with the exact-equality and constant-tolerance rules.
E8 adds the two remaining published families and recomputes the E4 rules on
the same ensembles and the same fault set:

  A(b)    exact equality of b-bit outputs (as E4)
  B       constant tolerance |z| > theta0 (as E4)
  C       drift-aware band |z - mu| > theta0 * rho_model(t) (this work, as E3/E4)
  C_g     the same band calibrated per checksum group at t0:
          |z| > k_C * q_g * rho_model(t), q_g the t0 RMS of group g
  M       mean-tracking time-aware threshold |z - mu_model(t)| > theta0_M
  O_g<s>  online variance-adaptive threshold, one estimator per checksum group:
          |z_n| > k * sqrt(v_{n-1}), v <- v + 2^-s (z^2 - v) every inference
  O_p<s>  the same with one estimator per array, fed by the mean of z^2 over the
          K groups of that array at each inference

z is the residual in the digitized readout domain (s * r with ratio readout).
k is calibrated at t0 as the (1 - FPR0) quantile of |z_n| / sqrt(v_{n-1}) over
the healthy ensemble, the same one-scalar calibration as theta0.

Drift over one estimator time constant (2^s inferences) is negligible against
the logarithmic time scale of drift, so the online estimators are evaluated on
the stationary inference stream of each time point: the EWMA starts at the
stream mean and runs `ewma_passes` circular passes over the 2000 inferences;
alarms are taken on the last pass. Fault probes reuse the E4 seed branch and
draw order, so the fault set is the E4 one. For a persistent fault, "onset"
evaluates the faulty stream against the estimator state of the healthy
stream (the first inferences after the fault appears), "steady" against the
estimator fed by the faulty stream itself (after many time constants).

Run from the project root:

    python experiments/E8_published_baselines.py --e3 results/E3/20260830_0124 \
        --e4 results/E4/20260830_0150             (formal, with regression gates)
    python experiments/E8_published_baselines.py --smoke    (pipeline check)

Outputs per run: results/E8/<runstamp>/{config.json, fpr_compare.csv,
discrimination.csv, gates.json}, figures/E8/*_<runstamp>.{pdf,png},
logs/E8_<runstamp>.log.
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
from E8_config import E8Config  # noqa: E402
from E3_validity_domain_scan import ENC_ID, build_ensemble, physics_params  # noqa: E402
from E4_baseline_comparison import detector_flags, pools_with_yck  # noqa: E402

FPR_FIELDS = ["encoding", "sigma_nu", "m", "c", "ratio_norm", "t_rel",
              "detector", "threshold", "fpr"]
DISC_FIELDS = ["encoding", "sigma_nu", "m", "c", "ratio_norm", "t_rel", "delta",
               "detector", "state", "p1", "p1_single", "fpr", "delta_p", "n_trials"]


# ---------------------------------------------------------------------------
# online estimator
# ---------------------------------------------------------------------------


def ewma_prev(x: np.ndarray, shift: int, passes: int) -> np.ndarray:
    """Causal EWMA state preceding each sample of a stationary stream.

    x [..., N], last axis in inference order. Returns v [..., N] with v[..., n]
    the state after samples 0..n-1 on the final circular pass, i.e. the value a
    hardware estimator holds when sample n arrives. The state starts at the
    stream mean, so the initial value enters the last pass with weight
    (1 - 2^-s)^((passes - 1) N).
    """
    alpha = 2.0 ** (-shift)
    n_s = x.shape[-1]
    v = x.mean(axis=-1)
    out = np.empty_like(x)
    for p in range(passes):
        last = p == passes - 1
        for n in range(n_s):
            if last:
                out[..., n] = v
            v = v + alpha * (x[..., n] - v)
    return out


def healthy_online(z: np.ndarray, shift: int, passes: int):
    """Healthy normalized statistics for both estimator scopes.

    z [A, N, K]. Returns (u_g, u_p, v_g, v_p): u = |z| / sqrt(v_prev) for the
    per-group (u_g, [A, N, K]) and per-array pooled (u_p, [A, N, K]) estimator,
    plus the estimator states v_g [A, K, N] and v_p [A, N].
    """
    z2 = z * z
    v_g = ewma_prev(np.transpose(z2, (0, 2, 1)), shift, passes)       # [A, K, N]
    v_p = ewma_prev(z2.mean(axis=2), shift, passes)                     # [A, N]
    u_g = np.abs(z) / np.sqrt(np.transpose(v_g, (0, 2, 1)))
    u_p = np.abs(z) / np.sqrt(v_p[:, :, None])
    return u_g, u_p, v_g, v_p


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results/E8")
    ap.add_argument("--e3", default=None, help="E3 official record dir (gate G4)")
    ap.add_argument("--e4", default=None, help="E4 official record dir (gates G1, G2)")
    args = ap.parse_args()

    cfg = E1Config()
    e3 = E3Config()
    e8 = E8Config()
    p = physics_params(cfg)
    t_grid = list(e3.t_rel_grid)
    sig_on = list(e8.sigma_nu_on)
    n_trials = e8.n_trials
    shifts = list(e8.ewma_shifts)
    if args.smoke:
        sig_on = [0.02]
        t_grid = [1.0, 1e2, 1e4]
        e3 = dataclasses.replace(e3, n_arrays=4, n_mvm=300)
        n_trials = 400

    paths = io_utils.resolve_paths(args.out, exp="E8", suffix="_smoke" if args.smoke else "")
    log = io_utils.get_logger("E8", paths.logs / f"E8_{paths.runstamp}.log")
    t_start = time.perf_counter()
    log.info("E8 run %s starting (smoke=%s)", paths.runstamp, args.smoke)
    io_utils.save_json(
        io_utils.config_snapshot(e8, vars(args), {
            "runstamp": paths.runstamp, "physics": dataclasses.asdict(cfg),
            "e3_seed_coords": {"seed_base": e3.seed_base, "n_arrays": e3.n_arrays,
                               "n_mvm": e3.n_mvm},
            "pcm_drift_version": pcm_drift.__version__,
        }),
        paths.run_dir / "config.json",
    )

    rows_fpr, rows_disc = [], []
    kcal = {}

    # ------------------------------------------------------------------
    # part 1: encoding b slice (E4 slice), all detectors, fault probes
    # ------------------------------------------------------------------
    confs = [(s, True) for s in sig_on]
    if not args.smoke:
        confs.append((e8.sigma_off_slice, False))
    m, c = e8.m, e8.c
    mi_g = e3.m_grid.index(m)
    ci_g = e3.c_grid.index(c)
    lam = c / m

    for sig, norm_on in confs:
        tc = time.perf_counter()
        si_g = e3.sigma_nu_grid.index(sig)
        arrays = build_ensemble("b", si_g, mi_g, ci_g, cfg, e3)
        r_off, yck_off, s_arr = pools_with_yck(arrays, t_grid, cfg)
        A = len(arrays)
        N = arrays[0][2].shape[0]
        K = arrays[0][1].K

        def z_at(ti):
            r = r_off[ti].astype(np.float64)
            return (s_arr[ti][:, None, None] * r) if norm_on else r

        # E4 calibrations (identical code path)
        s0 = s_arr[0][:, None, None] if norm_on else 1.0
        theta0 = float(np.quantile(np.abs(s0 * r_off[0].astype(np.float64)), 1.0 - e8.fpr0))
        mu_t0, sd_t0 = model.r_moments("b", m, 0.0, sig, p, norm_on, lam_scale=c)
        theta0_m = float(np.quantile(np.abs(z_at(0) - mu_t0), 1.0 - e8.fpr0))
        y_all_max = 0.0
        for a, (state, layout, X) in enumerate(arrays):
            M0, _ = encodings.effective_matrix(state, layout, 1.0, False, cfg)
            Y0 = X @ M0
            scale0 = s_arr[0, a] if norm_on else 1.0
            y_all_max = max(y_all_max, float(np.abs(scale0 * Y0).max()))
        FS = 2.0 * y_all_max
        q_of = {b: FS / (2 ** b) for b in e8.adc_bits}

        # online estimators: calibrate k at t0 (one scalar per estimator)
        z0 = z_at(0)
        k_of = {}
        for s_ in shifts:
            u_g, u_p, _, _ = healthy_online(z0, s_, e8.ewma_passes)
            k_of[f"O_g{s_}"] = float(np.quantile(u_g, 1.0 - e8.fpr0))
            k_of[f"O_p{s_}"] = float(np.quantile(u_p, 1.0 - e8.fpr0))
        # band with per-group calibration at t0 (one stored scalar per group)
        q_g = np.sqrt((z0 * z0).mean(axis=1))                   # [A, K]
        k_c = float(np.quantile(np.abs(z0) / q_g[:, None, :], 1.0 - e8.fpr0))
        kcal[f"b/{sig}/{int(norm_on)}"] = {"theta0": theta0, "theta0_M": theta0_m,
                                            "mu_model_t0": mu_t0, "k_C_g": k_c,
                                            "q_g_cv": float(q_g.std() / q_g.mean()), **k_of}
        log.info("config sigma=%.3g norm=%s: theta0=%.5f theta0_M=%.5f k=%s",
                 sig, "on" if norm_on else "off", theta0, theta0_m,
                 {kk: round(v, 5) for kk, v in k_of.items()})

        frng = np.random.default_rng(np.random.SeedSequence(
            [e3.seed_base, e8.fault_seed_branch, ENC_ID["b"], si_g, mi_g, ci_g, int(norm_on)]))

        for ti, t in enumerate(t_grid):
            L = float(np.log(t))
            mu_t, sd_t = model.r_moments("b", m, L, sig, p, norm_on, lam_scale=c)
            f_band = theta0 * (sd_t / sd_t0)
            z_t = z_at(ti)
            r_t = r_off[ti].astype(np.float64)
            y_t = yck_off[ti].astype(np.float64)
            s_t = s_arr[ti][:, None, None]

            fpr_now = {}
            # E4 detectors, identical code path (gate G1)
            static = [(f"A{b}", ("A", q_of[b])) for b in e8.adc_bits]
            static += [("B_const", ("T", theta0)), ("C_band", ("T", f_band))]
            for name, (kind, par) in static:
                if kind == "A":
                    fl = detector_flags("A", r_t, y_t, s_t, par, None, norm_on)
                else:
                    fl = detector_flags("B", r_t, y_t, s_t, None, par, norm_on)
                fpr_now[name] = float(fl.mean())
                rows_fpr.append(_frow("b", sig, m, c, norm_on, t, name, par, fpr_now[name]))
            # M: mean-tracking, constant width
            fpr_now["M_mean"] = float((np.abs(z_t - mu_t) > theta0_m).mean())
            rows_fpr.append(_frow("b", sig, m, c, norm_on, t, "M_mean", theta0_m, fpr_now["M_mean"]))
            rho_t = sd_t / sd_t0
            fpr_now["C_band_g"] = float((np.abs(z_t) > (k_c * rho_t) * q_g[:, None, :]).mean())
            rows_fpr.append(_frow("b", sig, m, c, norm_on, t, "C_band_g", k_c * rho_t, fpr_now["C_band_g"]))
            # O: online variance-adaptive
            states = {}
            for s_ in shifts:
                u_g, u_p, v_g, v_p = healthy_online(z_t, s_, e8.ewma_passes)
                states[s_] = (v_g, v_p)
                for nm, u in ((f"O_g{s_}", u_g), (f"O_p{s_}", u_p)):
                    fpr_now[nm] = float((u > k_of[nm]).mean())
                    rows_fpr.append(_frow("b", sig, m, c, norm_on, t, nm, k_of[nm], fpr_now[nm]))

            if t not in e8.probe_t:
                continue
            for d in e8.probe_deltas:
                # E4 draw order: aa, rows, cols, nn, br, sg
                aa = frng.integers(0, A, n_trials)
                rows_i = frng.integers(0, cfg.R, n_trials)
                cols = frng.integers(0, cfg.C, n_trials)
                nn = frng.integers(0, N, n_trials)
                br = frng.integers(0, 2, n_trials)
                sg = np.where(frng.integers(0, 2, n_trials) == 0, 1.0, -1.0)

                # per-trial faulty streams [n_trials, N]
                kk = cols // m
                rs = np.empty((n_trials, N))
                yk = np.empty((n_trials, N))
                xs = np.empty((n_trials, N))
                dm = np.empty(n_trials)
                sv = np.ones(n_trials)
                for a in range(A):
                    sel = np.flatnonzero(aa == a)
                    if sel.size == 0:
                        continue
                    state, layout, X = arrays[a]
                    j = cols[sel]
                    i = rows_i[sel]
                    pcol = np.where(br[sel] == 0, layout.pos_cols[j], layout.neg_cols[j])
                    bsign = np.where(br[sel] == 0, 1.0, -1.0)
                    g0 = state.g0[i, pcol]
                    nu = state.nu[i, pcol]
                    g_new = np.maximum(g0 + sg[sel] * d, cfg.g0_floor)
                    dm[sel] = bsign * (g_new - g0) * np.exp(-nu * L)
                    rs[sel] = r_t[a][:, kk[sel]].T
                    yk[sel] = y_t[a][:, kk[sel]].T
                    xs[sel] = X[:, i].T
                    if norm_on:
                        sv[sel] = s_arr[ti, a]
                shift = -lam * xs * dm[:, None]
                r1 = rs + shift
                z1 = (sv[:, None] * r1) if norm_on else r1
                z0s = (sv[:, None] * rs) if norm_on else rs
                idx = np.arange(n_trials)

                def add(name, state_, alarms, single=None):
                    p1v = float(alarms.mean())
                    rows_disc.append({
                        "encoding": "b", "sigma_nu": sig, "m": m, "c": c,
                        "ratio_norm": int(norm_on), "t_rel": t, "delta": d,
                        "detector": name, "state": state_, "p1": p1v,
                        "p1_single": ("" if single is None else single),
                        "fpr": fpr_now[name], "delta_p": p1v - fpr_now[name],
                        "n_trials": n_trials})

                # static detectors: all inferences of the faulty stream, plus
                # the E4 single-inference estimate at nn (gate G2)
                for name, (kind, par) in static:
                    if kind == "A":
                        y1 = (sv[:, None] * yk) if norm_on else yk
                        y2 = y1 - z1
                        al = np.round(y1 / par) != np.round(y2 / par)
                    else:
                        al = np.abs(z1) > par
                    add(name, "static", al, float(al[idx, nn].sum()) / n_trials)
                al = np.abs(z1 - mu_t) > theta0_m
                add("M_mean", "static", al, float(al[idx, nn].sum()) / n_trials)
                al = np.abs(z1) > ((k_c * rho_t) * q_g[aa, kk])[:, None]
                add("C_band_g", "static", al)

                # online detectors: onset (healthy estimator) and steady
                # (estimator fed by the faulty stream)
                for s_ in shifts:
                    v_g, v_p = states[s_]
                    kg, kp = k_of[f"O_g{s_}"], k_of[f"O_p{s_}"]
                    vg_h = v_g[aa, kk, :]                       # [n_trials, N]
                    vp_h = v_p[aa, :]                           # [n_trials, N]
                    add(f"O_g{s_}", "onset", np.abs(z1) / np.sqrt(vg_h) > kg)
                    vg_f = ewma_prev(z1 * z1, s_, e8.ewma_passes)
                    add(f"O_g{s_}", "steady", np.abs(z1) / np.sqrt(vg_f) > kg)
                    add(f"O_p{s_}", "onset", np.abs(z1) / np.sqrt(vp_h) > kp)
                    # pooled input with the faulty group's contribution replaced
                    zt2 = z_t * z_t                              # [A, N, K]
                    qa = zt2.mean(axis=2)[aa, :]                 # [n_trials, N]
                    qf = qa + (z1 * z1 - z0s * z0s) / K
                    vp_f = ewma_prev(qf, s_, e8.ewma_passes)
                    add(f"O_p{s_}", "steady", np.abs(z1) / np.sqrt(vp_f) > kp)
        log.info("config sigma=%.3g norm=%s done in %.1f s",
                 sig, "on" if norm_on else "off", time.perf_counter() - tc)

    # ------------------------------------------------------------------
    # part 2: bias mapping (encoding a), mean-tracking vs band vs constant
    # ------------------------------------------------------------------
    a_confs = [s for s in (sig_on if args.smoke else e3.sigma_nu_grid)]
    ma, ca = e8.enc_a_m, e8.enc_a_c
    for sig in a_confs:
        tc = time.perf_counter()
        si_g = e3.sigma_nu_grid.index(sig)
        arrays = build_ensemble("a", si_g, e3.m_grid.index(ma), e3.c_grid.index(ca), cfg, e3)
        r_off, _, s_arr = pools_with_yck(arrays, t_grid, cfg)
        z0 = s_arr[0][:, None, None] * r_off[0].astype(np.float64)
        theta0 = float(np.quantile(np.abs(z0), 1.0 - e8.fpr0))          # E3 definition
        mu0, sd0 = model.r_moments("a", ma, 0.0, sig, p, True, lam_scale=ca)
        theta0_m = float(np.quantile(np.abs(z0 - mu0), 1.0 - e8.fpr0))
        kcal[f"a/{sig}/1"] = {"theta0": theta0, "theta0_M": theta0_m, "mu_model_t0": mu0}
        for ti, t in enumerate(t_grid):
            L = float(np.log(t))
            mu_t, sd_t = model.r_moments("a", ma, L, sig, p, True, lam_scale=ca)
            f_band = theta0 * (sd_t / sd0)
            z_t = s_arr[ti][:, None, None] * r_off[ti].astype(np.float64)
            for name, thr, st in (("B_const", theta0, np.abs(z_t)),
                                  ("M_mean", theta0_m, np.abs(z_t - mu_t)),
                                  ("C_band", f_band, np.abs(z_t - mu_t))):
                rows_fpr.append(_frow("a", sig, ma, ca, True, t, name, thr,
                                      float((st > thr).mean())))
        log.info("encoding a sigma=%.3g done in %.1f s", sig, time.perf_counter() - tc)

    for name, rows, fields in (("fpr_compare.csv", rows_fpr, FPR_FIELDS),
                               ("discrimination.csv", rows_disc, DISC_FIELDS)):
        with open(paths.run_dir / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    io_utils.save_json(kcal, paths.run_dir / "calibration.json")
    log.info("written: fpr_compare.csv (%d rows), discrimination.csv (%d rows)",
             len(rows_fpr), len(rows_disc))

    gates = run_gates(rows_fpr, rows_disc, args, log)
    io_utils.save_json(gates, paths.run_dir / "gates.json")
    summarize(rows_fpr, rows_disc, t_grid, log)
    make_figures(rows_fpr, rows_disc, paths, log)
    log.info("E8 run %s complete in %.1f s; every number above traces to %s",
             paths.runstamp, time.perf_counter() - t_start, paths.run_dir)
    return 0


def _frow(enc, sig, m, c, norm_on, t, name, thr, fpr):
    return {"encoding": enc, "sigma_nu": sig, "m": m, "c": c, "ratio_norm": int(norm_on),
            "t_rel": t, "detector": name, "threshold": thr, "fpr": fpr}


# ---------------------------------------------------------------------------
# gates (spec section 5)
# ---------------------------------------------------------------------------


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def run_gates(rows_fpr, rows_disc, args, log):
    gates = {}
    # G3: mean tracking is the constant tolerance on the differential pair
    fb = {(r["sigma_nu"], r["ratio_norm"], r["t_rel"]): r["fpr"]
          for r in rows_fpr if r["encoding"] == "b" and r["detector"] == "B_const"}
    fm = {(r["sigma_nu"], r["ratio_norm"], r["t_rel"]): r["fpr"]
          for r in rows_fpr if r["encoding"] == "b" and r["detector"] == "M_mean"}
    db = {(r["sigma_nu"], r["ratio_norm"], r["t_rel"], r["delta"]): r["p1"]
          for r in rows_disc if r["detector"] == "B_const"}
    dmn = {(r["sigma_nu"], r["ratio_norm"], r["t_rel"], r["delta"]): r["p1"]
           for r in rows_disc if r["detector"] == "M_mean"}
    n_fpr = sum(fb[k] != fm[k] for k in fb)
    n_p1 = sum(db[k] != dmn[k] for k in db)
    gates["G3_M_equals_B_on_b"] = {"pass": n_fpr == 0 and n_p1 == 0,
                                   "fpr_mismatch": n_fpr, "p1_mismatch": n_p1}
    log.info("gate G3 (M == B on encoding b): %s (fpr mismatches %d, p1 mismatches %d)",
             "PASS" if gates["G3_M_equals_B_on_b"]["pass"] else "FAIL", n_fpr, n_p1)

    if args.e4:
        ref = _read_csv(Path(args.e4) / "fpr_compare.csv")
        mine = {(r["sigma_nu"], r["ratio_norm"], r["t_rel"], r["detector"]): r["fpr"]
                for r in rows_fpr if r["encoding"] == "b"}
        n, bad = 0, 0
        for r in ref:
            key = (float(r["sigma_nu"]), int(r["ratio_norm"]), float(r["t_rel"]), r["detector"])
            if key in mine:
                n += 1
                bad += mine[key] != float(r["fpr"])
        gates["G1_E4_fpr_reproduced"] = {"pass": n > 0 and bad == 0, "compared": n, "mismatch": bad}
        log.info("gate G1 (E4 FPR rows reproduced): %s (%d compared, %d mismatches)",
                 "PASS" if gates["G1_E4_fpr_reproduced"]["pass"] else "FAIL", n, bad)
        refd = _read_csv(Path(args.e4) / "discrimination.csv")
        mined = {(r["sigma_nu"], r["ratio_norm"], r["t_rel"], r["delta"], r["detector"]): r["p1_single"]
                 for r in rows_disc if r["state"] == "static"}
        n, bad = 0, 0
        for r in refd:
            key = (float(r["sigma_nu"]), int(r["ratio_norm"]), float(r["t_rel"]),
                   float(r["delta"]), r["detector"])
            if key in mined:
                n += 1
                bad += mined[key] != float(r["p1"])
        gates["G2_E4_p1_reproduced"] = {"pass": n > 0 and bad == 0, "compared": n, "mismatch": bad}
        log.info("gate G2 (E4 single-inference p1 reproduced): %s (%d compared, %d mismatches)",
                 "PASS" if gates["G2_E4_p1_reproduced"]["pass"] else "FAIL", n, bad)
    else:
        log.info("gates G1, G2 skipped: no --e4 record given")

    if args.e3:
        ref = _read_csv(Path(args.e3) / "fpr_backtest.csv")
        mine = {(r["sigma_nu"], r["t_rel"]): r["fpr"]
                for r in rows_fpr if r["encoding"] == "a" and r["detector"] == "C_band"}
        n, bad = 0, 0
        for r in ref:
            if r["encoding"] != "a" or int(r["m"]) != 16 or int(r["c"]) != 1 or int(r["ratio_norm"]) != 1:
                continue
            key = (float(r["sigma_nu"]), float(r["t_rel"]))
            if key in mine:
                n += 1
                bad += mine[key] != float(r["fpr"])
        gates["G4_E3_enc_a_band_reproduced"] = {"pass": n > 0 and bad == 0, "compared": n, "mismatch": bad}
        log.info("gate G4 (E3 encoding-a band FPR reproduced): %s (%d compared, %d mismatches)",
                 "PASS" if gates["G4_E3_enc_a_band_reproduced"]["pass"] else "FAIL", n, bad)
    else:
        log.info("gate G4 skipped: no --e3 record given")
    return gates


# ---------------------------------------------------------------------------
# summary lines and figures
# ---------------------------------------------------------------------------


def summarize(rows_fpr, rows_disc, t_grid, log):
    dets = ["A6", "A8", "A10", "B_const", "M_mean", "C_band", "C_band_g", "O_g6", "O_g10", "O_p6", "O_p10"]
    for enc in ("b", "a"):
        confs = sorted({(r["sigma_nu"], r["ratio_norm"]) for r in rows_fpr if r["encoding"] == enc})
        for sig, nm in confs:
            out = {}
            for d in dets:
                v = [r["fpr"] for r in rows_fpr if r["encoding"] == enc and r["sigma_nu"] == sig
                     and r["ratio_norm"] == nm and r["detector"] == d]
                if v:
                    out[d] = (round(min(v), 4), round(max(v), 4))
            log.info("FPR range over t, enc %s sigma=%.3g norm=%d: %s", enc, sig, nm, out)
    for r in rows_disc:
        pass
    keys = sorted({(r["sigma_nu"], r["ratio_norm"], r["t_rel"], r["delta"]) for r in rows_disc})
    for k in keys:
        sub = {f'{r["detector"]}/{r["state"]}': round(r["p1"], 4) for r in rows_disc
               if (r["sigma_nu"], r["ratio_norm"], r["t_rel"], r["delta"]) == k}
        log.info("p1 sigma=%.3g norm=%d t=%.0e delta=%.2f: %s", k[0], k[1], k[2], k[3], sub)


def make_figures(rows_fpr, rows_disc, paths, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {"A6": ("tab:red", ":"), "A8": ("tab:red", "--"), "A10": ("tab:red", "-."),
              "B_const": ("tab:orange", "-"), "C_band": ("tab:blue", "-"),
              "C_band_g": ("tab:cyan", "--"), "O_g6": ("tab:green", ":"), "O_g10": ("tab:green", "-"),
              "O_p6": ("tab:purple", ":"), "O_p10": ("tab:purple", "-")}
    panels = sorted({(r["sigma_nu"], r["ratio_norm"]) for r in rows_fpr if r["encoding"] == "b"},
                    key=lambda x: (-x[1], x[0]))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.2),
                             squeeze=False, sharey=True)
    for ax, (sig, nm) in zip(axes[0], panels):
        for det, (col, ls) in styles.items():
            pts = sorted((r["t_rel"], max(r["fpr"], 5e-4)) for r in rows_fpr
                         if r["encoding"] == "b" and r["sigma_nu"] == sig
                         and r["ratio_norm"] == nm and r["detector"] == det)
            if pts:
                ax.plot([q[0] for q in pts], [q[1] for q in pts], color=col, ls=ls,
                        lw=1.6 if det == "C_band" else 1.0, label=det)
        ax.axhline(0.01, color="k", lw=0.6, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(5e-4, 1.3)
        ax.set_xlabel("$t/t_0$")
        ax.set_title(f"$\\sigma_\\nu$={sig}, norm {'on' if nm else 'off'}", fontsize=9)
    axes[0][0].set_ylabel("false-alarm rate per inference")
    axes[0][0].legend(fontsize=6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(paths.figures / f"fig_e8_fpr_{paths.runstamp}.{ext}", dpi=200)
    plt.close(fig)

    if rows_disc:
        keys = sorted({(r["t_rel"], r["delta"]) for r in rows_disc
                       if r["sigma_nu"] == 0.02 and r["ratio_norm"] == 1})
        dets = [("C_band", "static"), ("C_band_g", "static"), ("O_g6", "onset"), ("O_g6", "steady"),
                ("O_g10", "onset"), ("O_g10", "steady"), ("O_p6", "onset"),
                ("O_p6", "steady"), ("O_p10", "onset"), ("O_p10", "steady")]
        fig, ax = plt.subplots(figsize=(7.0, 3.2))
        width = 0.09
        for di, (det, st) in enumerate(dets):
            vals = []
            for t, d in keys:
                v = [r["p1"] for r in rows_disc if r["sigma_nu"] == 0.02 and r["ratio_norm"] == 1
                     and r["t_rel"] == t and r["delta"] == d and r["detector"] == det
                     and r["state"] == st]
                vals.append(v[0] if v else 0.0)
            xs = np.arange(len(keys)) + (di - len(dets) / 2) * width
            ax.bar(xs, vals, width, label=f"{det} {st}", color=styles[det][0],
                   alpha=0.45 if st == "steady" else 0.95)
        ax.set_xticks(np.arange(len(keys)))
        ax.set_xticklabels([f"t=1e{int(np.log10(t))}\n$\\delta$={d}" for t, d in keys], fontsize=7)
        ax.set_ylabel("$p_1$ per inference")
        ax.set_title("$\\sigma_\\nu$=0.02, norm on: onset vs steady state of the online estimators",
                     fontsize=9)
        ax.legend(fontsize=6, ncol=3)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(paths.figures / f"fig_e8_masking_{paths.runstamp}.{ext}", dpi=200)
        plt.close(fig)
    log.info("figures written to %s", paths.figures)


if __name__ == "__main__":
    sys.exit(main())
