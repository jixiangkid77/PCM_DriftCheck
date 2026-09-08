#!/usr/bin/env python3
"""E6: hardware realization flow for the drift-aware checker.

Implements docs/E6_Design_Spec_2026-08-30.md (frozen). Phases:

  M    build the ensemble slice, calibrate theta0 per bit-width in the
       digitized domain, generate PWL coefficients (pwl_case.vh) and the
       eps_f(t) / predicted-dFPR budget curve (Prop 3);
  F    generate bit-true vectors from ensemble samples plus boundary cases,
       run iverilog on both testbenches (CSC = 1, CSC = 3, self-contained
       top) and require 100 percent match against the Python golden model;
  FPR  bit-true false-alarm backtest over nine decades per bit-width,
       including the wrong-rho control (analog rho at 8 bit) and t0 p1 spot
       checks (sign_flip, soft_abs 0.95) against the float analog reference;
  S    yosys + sky130_fd_sc_hd synthesis of all variants plus the latch-bank
       scale reference; netlists written for the KID-side OpenSTA run.

Run from the project root:

    python experiments/E6_checker_flow.py --smoke
    python experiments/E6_checker_flow.py
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[1]


def wsl_path(p):
    """D:\\a\\b -> /mnt/d/a/b (identity on posix paths)."""
    s = str(Path(p).resolve())
    if len(s) > 1 and s[1] == ":":
        return "/mnt/" + s[0].lower() + s[2:].replace("\\", "/")
    return s.replace("\\", "/")
sys.path.insert(0, str(ROOT / "src"))

import pcm_drift  # noqa: E402
from pcm_drift import checksum, encodings, io_utils, model  # noqa: E402

sys.path.insert(0, str(_HERE.parent))
from E1_config import E1Config  # noqa: E402
from E3_config import E3Config  # noqa: E402
from E6_config import E6Config  # noqa: E402
from E3_validity_domain_scan import ENC_ID, build_ensemble, physics_params  # noqa: E402
from E5_fault_class_coverage import class_dm  # noqa: E402

LMAX = float(np.log(1e8))


def rtl_r_abs(yck_code, s_sum, logm, csc):
    r = (yck_code.astype(np.int64) << logm) - csc * s_sum.astype(np.int64)
    return np.abs(r)


def alarm_of(rabs, f_int, fb):
    return (rabs.astype(np.int64) << fb) > f_int


def pwl_eval_int(base, dlt, seg, u, ub):
    return base[seg] + ((dlt[seg] * u) >> ub)


def seg_u_of_L(L, nsl, ub):
    nseg = 1 << nsl
    x = np.clip(L / LMAX, 0.0, 1.0 - 1e-12) * nseg
    seg = np.minimum(x.astype(int), nseg - 1)
    u = np.minimum(((x - seg) * (1 << ub)).astype(int), (1 << ub) - 1)
    return seg, u


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results/E6")
    ap.add_argument("--eda", choices=("auto", "native", "wsl"), default="auto",
                    help="how to invoke yosys: native subprocess or via wsl "
                         "(auto = wsl on Windows, native elsewhere)")
    args = ap.parse_args()
    args.eda_resolved = (("wsl" if sys.platform.startswith("win") else "native")
                         if args.eda == "auto" else args.eda)

    cfg = E1Config()
    e3 = E3Config()
    e6 = E6Config()
    p = physics_params(cfg)
    t_grid = list(e3.t_rel_grid)
    by_grid = list(e6.by_grid)
    n_vec = e6.n_vectors
    n_spot = e6.n_trials_spot
    variants = list(e6.synth_variants)
    if args.smoke:
        t_grid = [1.0, 1e2, 1e8]
        by_grid = [8]
        n_vec, n_spot = 400, 600
        variants = [(8, 4, 1), (8, 4, 3)]
        e3 = dataclasses.replace(e3, n_arrays=4, n_mvm=300)

    paths = io_utils.resolve_paths(args.out, exp="E6", suffix="_smoke" if args.smoke else "")
    log = io_utils.get_logger("E6", paths.logs / f"E6_{paths.runstamp}.log")
    t0c = time.perf_counter()
    log.info("E6 run %s starting (smoke=%s)", paths.runstamp, args.smoke)
    io_utils.save_json(io_utils.config_snapshot(e6, vars(args), {
        "runstamp": paths.runstamp, "physics": dataclasses.asdict(cfg),
        "pcm_drift_version": pcm_drift.__version__}), paths.run_dir / "config.json")

    # ---------------- ensemble + digitized pools (shared by M/F/FPR) --------
    sig, m, cval = e6.sigma_nu, e6.m, e6.c
    logm = int(np.log2(m))
    si_g, mi_g, ci_g = (e3.sigma_nu_grid.index(sig), e3.m_grid.index(m),
                        e3.c_grid.index(cval))
    arrays = build_ensemble("b", si_g, mi_g, ci_g, cfg, e3)
    A = len(arrays)
    N = arrays[0][2].shape[0]
    K = arrays[0][1].K
    C = arrays[0][1].C

    y_max0 = 0.0
    Y_on0 = []
    for a, (state, layout, X) in enumerate(arrays):
        M0, _ = encodings.effective_matrix(state, layout, 1.0, False, cfg)
        s0 = encodings._ratio_scale(state, 1.0, True)
        Y = (s0 * (X @ M0)).astype(np.float32)
        Y_on0.append(Y)
        y_max0 = max(y_max0, float(np.abs(Y).max()))
    FS = 2.0 * y_max0
    q_of = {b: FS / (2 ** b) for b in by_grid}
    code_max = {b: 2 ** (b - 1) - 1 for b in by_grid}
    log.info("ensemble ready: FS=%.3f, q=%s", FS,
             {b: round(v, 4) for b, v in q_of.items()})

    # digitized residual pools per bit-width and float analog pools
    r_int = {b: np.empty((len(t_grid), A, N, K), dtype=np.int64) for b in by_grid}
    r_flt = np.empty((len(t_grid), A, N, K), dtype=np.float64)
    for a, (state, layout, X) in enumerate(arrays):
        for ti, t in enumerate(t_grid):
            if ti == 0:
                Y = Y_on0[a].astype(np.float64)
            else:
                M0, _ = encodings.effective_matrix(state, layout, float(t), False, cfg)
                Y = encodings._ratio_scale(state, float(t), True) * (X @ M0)
            r_flt[ti, a] = checksum.residuals(Y, layout)
            for b in by_grid:
                codes = np.clip(np.round(Y / q_of[b]), -code_max[b] - 1,
                                code_max[b]).astype(np.int64)
                yck = codes[:, C:C + K]
                s_sum = codes[:, :C].reshape(N, K, m).sum(axis=2)
                r_int[b][ti, a] = (yck << logm) - s_sum
    theta0_code = {b: float(np.quantile(np.abs(r_int[b][0]), 1.0 - e6.fpr0))
                   for b in by_grid}
    theta0_flt = float(np.quantile(np.abs(r_flt[0]), 1.0 - e6.fpr0))
    # one-time sigma_an(L) interpolant: r_moments is smooth in L and costly
    Lgrid = np.linspace(0.0, LMAX, 49)
    sd_grid = np.array([model.r_moments("b", m, float(L), sig, p, True,
                                        lam_scale=cval)[1] for L in Lgrid])
    sd_an = lambda L: float(np.interp(L, Lgrid, sd_grid))
    sd0_an = sd_an(0.0)

    def rho_dig(b, L):
        return (model.sigma_digitized(sd_an(L), q_of[b], m)
                / model.sigma_digitized(sd0_an, q_of[b], m))

    # ---------------- Phase M: PWL coefficients + budget curve --------------
    nsl, ub, fb = e6.n_seg_log, e6.u_bits, e6.f_frac_bits
    nseg = 1 << nsl
    WR = e6.by_main + logm + 2 + fb
    bmain = e6.by_main if e6.by_main in by_grid else by_grid[0]
    f_exact = lambda L: theta0_code[bmain] * rho_dig(bmain, L) * (1 << fb)
    nodes = np.linspace(0.0, LMAX, nseg + 1)
    base = np.round([f_exact(L) for L in nodes[:-1]]).astype(np.int64)
    ends = np.round([f_exact(L) for L in nodes[1:]]).astype(np.int64)
    dlt = ends - base
    assert base.max() < 2 ** WR and dlt.max() < 2 ** (WR - 1) and dlt.min() >= 0
    with open(ROOT / "rtl" / "pwl_case.vh", "w", newline="\n") as fh:
        fh.write("// GENERATED by E6_checker_flow.py Phase M -- do not hand-edit.\n"
                 "// Authoritative coefficient record: results/E6/<runstamp>/pwl_params.json.\n")
        for k in range(nseg):
            fh.write(f"            {nsl}'d{k}: begin base = {WR}'d{base[k]}; "
                     f"dlt = {WR}'d{dlt[k]}; end\n")
        fh.write(f"            default: begin base = {WR}'d{base[-1]}; "
                 f"dlt = {WR}'d0; end\n")
    Ld = np.linspace(0.0, LMAX, 481)
    segd, ud = seg_u_of_L(Ld, nsl, ub)
    f_hw = pwl_eval_int(base, dlt, segd, ud, ub).astype(float)
    f_ex = np.array([f_exact(L) for L in Ld])  # both in 1/2^fb code units
    eps = np.abs(f_hw - f_ex) / f_ex
    z0 = 2.5758293035489004
    phi = float(np.exp(-z0 * z0 / 2) / np.sqrt(2 * np.pi))
    dfpr_pred = 2 * phi * z0 * eps
    io_utils.save_json({"FS": FS, "q": q_of, "theta0_code": theta0_code,
                        "n_seg": nseg, "u_bits": ub, "WR": WR, "f_frac_bits": fb,
                        "base": base.tolist(), "dlt": dlt.tolist(),
                        "eps_max": float(eps.max()),
                        "dfpr_pred_max": float(dfpr_pred.max())},
                       paths.run_dir / "pwl_params.json")
    Ld_keep = Ld
    with open(paths.run_dir / "eps_curve.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["L", "t_rel", "f_exact", "f_hw", "eps_f", "dfpr_pred"])
        for i in range(len(Ld)):
            w.writerow([Ld[i], float(np.exp(Ld[i])), f_ex[i], f_hw[i],
                        eps[i], dfpr_pred[i]])
    gate_M = eps.max() <= e6.eps_f_target
    log.info("Phase M: eps_f max %.4f (target %.3f, budget %.3f) -> %s; "
             "predicted dFPR max %.5f", eps.max(), e6.eps_f_target,
             e6.kappa_p * e6.fpr0 / (2 * phi * z0), "PASS" if gate_M else "FAIL",
             dfpr_pred.max())
    if not gate_M:
        raise AssertionError("Phase M PWL target missed")

    # ---------------- Phase F: bit-true vectors + iverilog ------------------
    rng = np.random.default_rng(e6.vec_seed)
    b = bmain
    WRb = b + logm + 2

    def pack_vec_c(yck, ys, f_int, csc):
        s_sum = int(ys.sum())
        rabs = int(rtl_r_abs(np.array([yck]), np.array([s_sum]), logm, csc)[0])
        alarm = int((rabs << fb) > f_int)
        word = alarm
        word = (word << WRb) | rabs
        word = (word << (WRb + fb)) | f_int
        word = (word << b) | (int(yck) & ((1 << b) - 1))
        for v in ys[::-1]:
            word = (word << b) | (int(v) & ((1 << b) - 1))
        return word, alarm, rabs

    def sample_codes(n):
        out = []
        for _ in range(n):
            a = rng.integers(0, A)
            ti = rng.integers(0, len(t_grid))
            n_in = rng.integers(0, N)
            k = rng.integers(0, K)
            state, layout, X = arrays[a]
            t = t_grid[ti]
            M0, _ = encodings.effective_matrix(state, layout, float(t), False, cfg)
            row = (encodings._ratio_scale(state, float(t), True)
                   * (X[n_in] @ M0))
            codes = np.clip(np.round(row / q_of[b]), -code_max[b] - 1,
                            code_max[b]).astype(np.int64)
            out.append((codes[C + k], codes[k * m:(k + 1) * m], ti))
        return out

    # boundary cases: adjust y_ck so |r| hits f-1, f, f+1 exactly
    for csc, fname in ((1, "vectors_c1.hex"), (3, "vectors_c3.hex")):
        words = []
        samp = sample_codes(n_vec - 6)
        for yck, ys, ti in samp:
            L = float(np.log(t_grid[ti]))
            f_int = int(round(theta0_code[b] * rho_dig(b, L) * (1 << fb)))
            words.append(pack_vec_c(int(yck), ys, f_int, csc)[0])
        yck0, ys0, _ = samp[0]
        f_int = int(round(theta0_code[b])) << fb
        s_sum = csc * int(ys0.sum())
        for target in ((f_int >> fb) - 1, f_int >> fb, (f_int >> fb) + 1):
            for sgn in (1, -1):
                yck_t = (sgn * target + s_sum) >> logm
                yck_t = int(np.clip(yck_t, -code_max[b] - 1, code_max[b]))
                words.append(pack_vec_c(yck_t, ys0, f_int, csc)[0])
        hexw = (len(bin(max(words))) - 2 + 3) // 4
        with open(ROOT / "tb" / fname, "w") as fh:
            for wd in words:
                fh.write(f"{wd:0{hexw}x}\n")
    # self-contained top vectors: random (seg, u) + ensemble codes
    words = []
    samp = sample_codes(n_vec)
    for yck, ys, ti in samp:
        seg = int(rng.integers(0, nseg))
        u = int(rng.integers(0, 1 << ub))
        f_int = int(pwl_eval_int(base, dlt, np.array([seg]), np.array([u]), ub)[0])
        s_sum = int(ys.sum())
        rabs = int(rtl_r_abs(np.array([int(yck)]), np.array([s_sum]), logm, 1)[0])
        alarm = int((rabs << fb) > f_int)
        word = alarm
        word = (word << WRb) | rabs
        word = (word << (WRb + fb)) | f_int
        word = (word << nsl) | seg
        word = (word << ub) | u
        word = (word << b) | (int(yck) & ((1 << b) - 1))
        for v in ys[::-1]:
            word = (word << b) | (int(v) & ((1 << b) - 1))
        words.append(word)
    hexw = (len(bin(max(words))) - 2 + 3) // 4
    with open(ROOT / "tb" / "vectors_top.hex", "w") as fh:
        for wd in words:
            fh.write(f"{wd:0{hexw}x}\n")

    def run_iverilog(tag, tb, extra, nv):
        out = ROOT / "tb" / f"sim_{tag}.vvp"
        cc = subprocess.run(
            ["iverilog", "-g2005", "-o", str(out), f"-DNVEC={nv}", *extra,
             "-I", str(ROOT / "rtl"), str(ROOT / "tb" / tb),
             str(ROOT / "rtl" / "drift_checker.v"),
             str(ROOT / "rtl" / "pwl_band.v"),
             str(ROOT / "rtl" / "checker_top.v")],
            capture_output=True, text=True)
        if cc.returncode != 0:
            raise RuntimeError(f"iverilog compile {tag}: {cc.stderr[-800:]}")
        rr = subprocess.run(["vvp", str(out)], cwd=str(ROOT / "tb"),
                            capture_output=True, text=True)
        mm = re.search(r"RESULT (\d+) mismatches of (\d+)", rr.stdout)
        if not mm:
            raise RuntimeError(f"vvp {tag}: {rr.stdout[-800:]}")
        return int(mm.group(1)), int(mm.group(2))

    res_f = {}
    res_f["c1"] = run_iverilog("c1", "tb_drift_checker.v", [], n_vec)
    res_f["c3"] = run_iverilog("c3", "tb_drift_checker.v", ["-DCSC3"], n_vec)
    res_f["top"] = run_iverilog("top", "tb_checker_top.v", [], n_vec)
    for tag, (bad, tot) in res_f.items():
        log.info("Phase F %s: %d mismatches of %d -> %s", tag, bad, tot,
                 "PASS" if bad == 0 else "FAIL")
    if any(bad for bad, _ in res_f.values()):
        raise AssertionError(f"Phase F FAILED: {res_f}")

    # ---------------- Phase FPR: bit-true backtest + p1 spots ---------------
    rows_fpr = []
    for bb in by_grid:
        for ti, t in enumerate(t_grid):
            L = float(np.log(t))
            f_flt_ref = theta0_flt * (sd_an(L) / sd0_an)
            fpr_ref = float((np.abs(r_flt[ti]) > f_flt_ref).mean())
            if bb == bmain:
                seg, u = seg_u_of_L(np.array([L]), nsl, ub)
                f_int = int(pwl_eval_int(base, dlt, seg, u, ub)[0])
                mode = "pwl"
            else:
                f_int = int(round(theta0_code[bb] * rho_dig(bb, L) * (1 << fb)))
                mode = "register"
            fpr_bt = float(alarm_of(np.abs(r_int[bb][ti]), f_int, fb).mean())
            f_wrong = int(round(theta0_code[bb] * (sd_an(L) / sd0_an) * (1 << fb)))
            fpr_wrong = float(alarm_of(np.abs(r_int[bb][ti]), f_wrong, fb).mean())
            rows_fpr.append({"by": bb, "t_rel": t, "mode": mode,
                             "f_int": f_int, "fpr_bittrue": fpr_bt,
                             "fpr_float_ref": fpr_ref,
                             "fpr_wrong_rho": fpr_wrong,
                             "diff": fpr_bt - fpr_ref})
    with open(paths.run_dir / "fpr_bittrue.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_fpr[0].keys()))
        w.writeheader()
        w.writerows(rows_fpr)
    diffs_g = [abs(r["diff"]) for r in rows_fpr if r["by"] in e6.gate_by]
    diffs_c = [(r["by"], r["t_rel"], r["diff"]) for r in rows_fpr
               if r["by"] not in e6.gate_by and abs(r["diff"]) > e6.tol_fpr_abs]
    gate_fpr = max(diffs_g) <= e6.tol_fpr_abs
    wrong8 = [r["fpr_wrong_rho"] for r in rows_fpr
              if r["by"] == bmain and r["t_rel"] >= 1e6]
    if args.smoke:
        # The 0.005 gate is calibrated for the full-run sample size; at smoke
        # scale the paired MC noise alone can cross it, so smoke only reports.
        log.info("Phase FPR (smoke, informational): |bittrue-float| max %.4f on "
                 "axis %s; statistical gate %.3f applies to the full run only; "
                 "wrong-rho control at by=%d, t>=1e6: %s", max(diffs_g),
                 e6.gate_by, e6.tol_fpr_abs, bmain,
                 [f"{v:.5f}" for v in wrong8])
    else:
        log.info("Phase FPR: |bittrue-float| max %.4f on gate axis %s (gate %.3f) -> %s; "
                 "wrong-rho control at by=%d, t>=1e6: %s", max(diffs_g), e6.gate_by,
                 e6.tol_fpr_abs, "PASS" if gate_fpr else "FAIL", bmain,
                 [f"{v:.5f}" for v in wrong8])
    if diffs_c:
        log.info("Phase FPR characterization rows outside tol (expected, "
                 "scale-banding precondition violated): %s",
                 [(b, f"{t:.0e}", f"{d:+.4f}") for b, t, d in diffs_c])
    if not gate_fpr and not args.smoke:
        raise AssertionError("Phase FPR gate FAILED")

    rows_spot = []
    frng = np.random.default_rng(e6.vec_seed + 1)
    state0 = [(arrays[a][0], arrays[a][1], arrays[a][2]) for a in range(A)]
    for kind, d in e6.spot_classes:
        aa = frng.integers(0, A, n_spot)
        ii = frng.integers(0, cfg.R, n_spot)
        jj = frng.integers(0, C, n_spot)
        nn = frng.integers(0, N, n_spot)
        br = frng.integers(0, 2, n_spot)
        sg = np.where(frng.integers(0, 2, n_spot) == 0, 1.0, -1.0)
        hits = {bb: 0 for bb in by_grid}
        hits_f = 0
        for a in range(A):
            sel = aa == a
            if not sel.any():
                continue
            state, layout, X = state0[a]
            i, j, n_in = ii[sel], jj[sel], nn[sel]
            k = j // m
            s0 = encodings._ratio_scale(state, 1.0, True)
            dm_on = s0 * class_dm("b", kind, d, state, layout, cfg, p,
                                  i, j, br[sel], sg[sel], 0.0)
            dy = X[n_in, i] * dm_on
            yj = Y_on0[a][n_in, j].astype(np.float64)
            r0f = r_flt[0, a, n_in, k]
            hits_f += int((np.abs(r0f - dy / m) > theta0_flt).sum())
            for bb in by_grid:
                old_c = np.clip(np.round(yj / q_of[bb]), -code_max[bb] - 1,
                                code_max[bb]).astype(np.int64)
                new_c = np.clip(np.round((yj + dy) / q_of[bb]),
                                -code_max[bb] - 1, code_max[bb]).astype(np.int64)
                r1 = r_int[bb][0, a, n_in, k] - (new_c - old_c)
                f_int = int(round(theta0_code[bb] * (1 << fb)))
                hits[bb] += int(alarm_of(np.abs(r1), f_int, fb).sum())
        p1f = hits_f / n_spot
        for bb in by_grid:
            rows_spot.append({"fault": kind, "delta": d, "by": bb,
                              "p1_bittrue": hits[bb] / n_spot,
                              "p1_float": p1f,
                              "diff": hits[bb] / n_spot - p1f})
    with open(paths.run_dir / "p1_spot.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_spot[0].keys()))
        w.writeheader()
        w.writerows(rows_spot)
    for r in rows_spot:
        log.info("Phase FPR spot %s by=%d: bittrue %.3f vs float %.3f",
                 r["fault"], r["by"], r["p1_bittrue"], r["p1_float"])

    # ---------------- Phase S: yosys synthesis ------------------------------
    # Scripts use paths relative to the project root and are written to
    # synth/gen_<tag>.ys; yosys runs with cwd=ROOT either natively or through
    # `wsl --cd` (args.eda_resolved), so no other path translation is needed.
    lib = e6.liberty.replace("\\", "/")
    if not (ROOT / e6.liberty).exists():
        raise FileNotFoundError(f"liberty not found: {ROOT / e6.liberty}")
    areas = {}
    (ROOT / "synth" / "out").mkdir(exist_ok=True)

    def run_yosys(script_text, tag):
        ys = ROOT / "synth" / f"gen_{tag}.ys"
        with open(ys, "w", newline="\n") as fh:
            fh.write(script_text)
        argv = ["yosys", "-QT", "-s", f"synth/gen_{tag}.ys"]
        if args.eda_resolved == "wsl":
            argv = ["wsl", "--cd", wsl_path(ROOT)] + argv
        rr = subprocess.run(argv, cwd=str(ROOT), capture_output=True, text=True)
        if rr.returncode != 0:
            raise RuntimeError(f"yosys {tag}: {(rr.stderr or rr.stdout)[-600:]}")
        return rr

    def synth(top, params, tag):
        chp = " ".join(f"-set {k} {v}" for k, v in params.items())
        script = ("read_verilog rtl/drift_checker.v "
                  "rtl/pwl_band.v rtl/checker_top.v\n"
                  + (f"chparam {chp} {top}\n" if params else "")
                  + f"hierarchy -top {top}\nsynth -top {top}\n"
                  f"dfflibmap -liberty {lib}\nabc -liberty {lib}\n"
                  f"opt_clean\nstat -liberty {lib}\n"
                  f"write_verilog -noattr synth/out/{tag}.v\n")
        rr = run_yosys(script, tag)
        cells = re.findall(r"Number of cells:\s+(\d+)", rr.stdout)
        area = re.findall(r"Chip area for (?:top )?module.*?:\s+([0-9.]+)", rr.stdout)
        return {"cells": int(cells[-1]) if cells else None,
                "area_um2": float(area[-1]) if area else None}

    for (BY, LOGM, CSC) in variants:
        tag = f"drift_checker_{BY}_{LOGM}_c{CSC}"
        areas[tag] = synth("drift_checker",
                           {"BY": BY, "LOGM": LOGM, "CSC": CSC}, tag)
        log.info("Phase S %s: %s", tag, areas[tag])
    tag = "checker_top_8_4"
    areas[tag] = synth("checker_top", {"BY": bmain, "LOGM": logm}, tag)
    log.info("Phase S %s: %s", tag, areas[tag])
    # scale reference: an m x BY output latch bank
    ref_v = ROOT / "rtl" / "latch_bank.v"
    with open(ref_v, "w", newline="\n") as fh:
        fh.write(
            "// E6 scale-reference module (spec section 1, decision 4): m-column output\n"
            "// latch bank, synthesized only as an area yardstick; not part of the checker.\n"
            "module latch_bank #(parameter BY=8, LOGM=4)(input wire clk,\n"
            "  input wire [(1<<LOGM)*BY-1:0] d, output reg [(1<<LOGM)*BY-1:0] qo);\n"
            "  always @(posedge clk) qo <= d;\nendmodule\n")
    script = (f"read_verilog rtl/latch_bank.v\nchparam -set BY {bmain} "
              f"-set LOGM {logm} latch_bank\nsynth -top latch_bank\n"
              f"dfflibmap -liberty {lib}\nabc -liberty {lib}\n"
              f"stat -liberty {lib}\n")
    rr = run_yosys(script, "latch_bank_ref")
    area = re.findall(r"Chip area for (?:top )?module.*?:\s+([0-9.]+)", rr.stdout)
    areas["latch_bank_ref"] = {"area_um2": float(area[-1]) if area else None}
    log.info("Phase S latch_bank_ref: %s", areas["latch_bank_ref"])
    io_utils.save_json(areas, paths.run_dir / "area_stat.json")

    make_figures(rows_fpr, rows_spot, Ld_keep, dfpr_pred, e6, by_grid, paths, log)

    log.info("E6 run %s complete in %.1f s; every number above traces to %s",
             paths.runstamp, time.perf_counter() - t0c, paths.run_dir)
    return 0


def make_figures(rows_fpr, rows_spot, Ld, dfpr_pred, e6, by_grid, paths, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.fill_between([1, 1e8], -e6.tol_fpr_abs, e6.tol_fpr_abs,
                    color="tab:blue", alpha=0.10, label="budget $\\pm\\kappa' FPR_0$")
    ax.plot(np.exp(Ld), dfpr_pred, "k-", lw=1.0, label="Prop 3 predicted (PWL $\\epsilon_f$)")
    ax.plot(np.exp(Ld), -dfpr_pred, "k-", lw=1.0)
    mk = {8: ("o", "tab:blue"), 10: ("s", "tab:green"), 6: ("x", "tab:red")}
    for bb in by_grid:
        pts = sorted((r["t_rel"], r["diff"]) for r in rows_fpr if r["by"] == bb)
        m_, c_ = mk.get(bb, ("d", "gray"))
        ax.plot([q[0] for q in pts], [q[1] for q in pts], m_, color=c_, ms=4,
                ls="none", label=f"bit-true $b_y$={bb}")
    ax.set_xscale("log")
    ax.set_xlabel("$t/t_0$")
    ax.set_ylabel("FPR(bit-true) - FPR(float ref)")
    ax.set_ylim(-0.012, 0.028)
    ax.legend(fontsize=6.5)
    ax.set_title("hardware band vs budget ($\\sigma_\\nu$=0.02, m=16)", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(paths.figures / f"fig_e6_fpr_budget_{paths.runstamp}.{ext}", dpi=200)
    plt.close(fig)

    if rows_spot:
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        classes = sorted({r["fault"] for r in rows_spot})
        width = 0.32
        for ci, cl in enumerate(classes):
            sub = sorted((r["by"], r["p1_bittrue"]) for r in rows_spot if r["fault"] == cl)
            xs = np.arange(len(sub)) + (ci - 0.5) * width
            ax.bar(xs, [q[1] for q in sub], width, label=cl)
            pf = [r["p1_float"] for r in rows_spot if r["fault"] == cl][0]
            ax.axhline(pf, color=f"C{ci}", ls="--", lw=0.9)
        bys = sorted({r["by"] for r in rows_spot})
        ax.set_xticks(np.arange(len(bys)))
        ax.set_xticklabels([f"$b_y$={b}" for b in bys])
        ax.set_ylabel("$p_1$ at $t_0$")
        ax.set_title("digitization cost of early detection (dashed: float)", fontsize=8.5)
        ax.legend(fontsize=7)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(paths.figures / f"fig_e6_by_tradeoff_{paths.runstamp}.{ext}", dpi=200)
        plt.close(fig)
    log.info("figures written to %s", paths.figures)


if __name__ == "__main__":
    sys.exit(main())
