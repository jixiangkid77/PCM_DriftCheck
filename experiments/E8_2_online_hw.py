#!/usr/bin/env python3
"""E8.2: hardware cost of the online variance-adaptive checkers (cost reference).

Plan: docs/E8.2_Plan_2026-10-11.md. The online estimators of E8 (O_g, O_p)
are written as RTL with the datapath of drift_checker up to |r|, verified bit
for bit against a fixed-point reference with Icarus Verilog, and synthesized
with the E6 flow (yosys 0.33, sky130_fd_sc_hd typical corner) so their area
can be set against the band checker of Table I. k^2 is fixed at design time
(K2 / 2^KF, from the E8 calibration), the cheapest form of the alternative.
Amendment A-1 adds online_checker_l1, which tracks E|r| instead of E r^2 and
so needs no squarer (hardware cost only; its detection is not simulated).

Run from the project root (EDA tools on PATH, liberty in synth/):

    python experiments/E8_2_online_hw.py --e8 results/E8/20261011_0440

Outputs per run: results/E8.2/<runstamp>/{config.json, area.json, bittrue.json},
tb/vectors_online_*.hex, synth/gen_online_*.ys, synth/out/online_*.v,
logs/E8.2_<runstamp>.log.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcm_drift import io_utils  # noqa: E402

BY, LOGM, LOGK, KF = 8, 4, 3, 6
M, K = 1 << LOGM, 1 << LOGK
WR = BY + LOGM + 2
GE_UM2 = 3.7536          # sky130_fd_sc_hd__nand2_1 area, as Table I
N_VEC = 4000
SLICE = "b/0.02/1"       # E8 calibration key: encoding b, sigma_nu 0.02, ratio readout
LIB = "synth/sky130_fd_sc_hd__tt_025C_1v80.lib"


def k2_int(k: float) -> int:
    return int(round(k * k * (1 << KF)))


def sample_codes(rng, n, k_groups):
    """Column codes with small residuals and occasional large deviations."""
    y = np.clip(np.rint(rng.normal(0.0, 30.0, size=(n, k_groups, M))), -128, 127).astype(np.int64)
    e = np.rint(rng.normal(0.0, 1.5, size=(n, k_groups))).astype(np.int64)
    spike = rng.random((n, k_groups)) < 0.03
    e = e + spike * rng.choice([-1, 1], size=(n, k_groups)) * rng.integers(5, 21, size=(n, k_groups))
    yck = np.clip(np.rint(y.sum(axis=2) / M).astype(np.int64) + e, -128, 127)
    return yck, y


def rabs_of(yck, y):
    r = (yck << LOGM) - y.sum(axis=-1)
    return np.abs(r)


def ref_group(rabs, s, k2, v0):
    v, al = int(v0), []
    for ra in rabs:
        q = int(ra) * int(ra)
        al.append(int((q << (s + KF)) > k2 * v))
        v = v + q - (v >> s)
    return al


def ref_group_l1(rabs, s, k1, m0):
    mv, al = int(m0), []
    for ra in rabs:
        al.append(int((int(ra) << (s + KF)) > k1 * mv))
        mv = mv + int(ra) - (mv >> s)
    return al


def ref_array(rabs_k, s, k2, v0):
    v, al = int(v0), []
    for row in rabs_k:
        qs = [int(ra) * int(ra) for ra in row]
        al.append([int((q << (s + KF)) > k2 * v) for q in qs])
        v = v + (sum(qs) >> LOGK) - (v >> s)
    return al


def to_hex_lines(words):
    hexw = (max(words).bit_length() + 3) // 4
    return [f"{w:0{hexw}x}" for w in words]


def pack_signed(val, width):
    return int(val) & ((1 << width) - 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/E8.2")
    ap.add_argument("--e8", required=True, help="E8 formal record (calibration.json)")
    args = ap.parse_args()

    paths = io_utils.resolve_paths(args.out, exp="E8.2")
    log = io_utils.get_logger("E8.2", paths.logs / f"E8.2_{paths.runstamp}.log")
    t_start = time.perf_counter()
    cal = json.loads((Path(args.e8) / "calibration.json").read_text(encoding="utf-8"))[SLICE]
    variants = []
    for s in (6, 10):
        variants.append(("online_checker", s, k2_int(cal[f"O_g{s}"]), cal[f"O_g{s}"]))
        variants.append(("online_array", s, k2_int(cal[f"O_p{s}"]), cal[f"O_p{s}"]))
        # amendment A-1: E|r| tracker, k1 = k * sqrt(pi/2) (Gaussian equivalence)
        k1 = cal[f"O_g{s}"] * float(np.sqrt(np.pi / 2.0))
        variants.append(("online_checker_l1", s, int(round(k1 * (1 << KF))), k1))
    io_utils.save_json({"runstamp": paths.runstamp, "e8": args.e8, "slice": SLICE,
                        "BY": BY, "LOGM": LOGM, "LOGK": LOGK, "KF": KF,
                        "variants": variants, "liberty": LIB, "n_vec": N_VEC},
                       paths.run_dir / "config.json")
    log.info("E8.2 run %s starting; variants %s", paths.runstamp, variants)
    rng = np.random.default_rng(20261011)

    # ---------------- bit-true vectors and Icarus runs ----------------
    bittrue = {}
    for top, s, k2, _ in variants:
        tag = f"{top}_s{s}"
        if top in ("online_checker", "online_checker_l1"):
            yck, y = sample_codes(rng, N_VEC, 1)
            ra = rabs_of(yck[:, 0], y[:, 0, :])
            if top == "online_checker":
                v0 = int(np.mean((ra[:256].astype(np.int64) ** 2)) * (1 << s))
                al = ref_group(ra, s, k2, v0)
            else:
                v0 = int(np.mean(ra[:256].astype(np.int64)) * (1 << s))
                al = ref_group_l1(ra, s, k2, v0)
            words = []
            for n in range(N_VEC):
                wv = al[n]
                wv = (wv << WR) | int(ra[n])
                wv = (wv << BY) | pack_signed(yck[n, 0], BY)
                yf = 0
                for j in range(M):
                    yf |= pack_signed(y[n, 0, j], BY) << (j * BY)
                wv = (wv << (M * BY)) | yf
                words.append(wv)
            tb = f"tb_{top}.v"
        else:
            yck, y = sample_codes(rng, N_VEC, K)
            ra = rabs_of(yck, y)                                  # [N, K]
            v0 = int(np.mean(ra[:256].astype(np.int64) ** 2) * (1 << s))
            al_k = ref_array(ra, s, k2, v0)
            al = [sum(b << k for k, b in enumerate(row)) for row in al_k]
            words = []
            for n in range(N_VEC):
                wv = al[n]
                yc = 0
                for k in range(K):
                    yc |= pack_signed(yck[n, k], BY) << (k * BY)
                wv = (wv << (K * BY)) | yc
                yf = 0
                for k in range(K):
                    for j in range(M):
                        yf |= pack_signed(y[n, k, j], BY) << ((k * M + j) * BY)
                wv = (wv << (K * M * BY)) | yf
                words.append(wv)
            tb = "tb_online_array.v"
        vec = ROOT / "tb" / f"vectors_{tag}.hex"
        vec.write_text("\n".join(to_hex_lines(words)) + "\n", encoding="utf-8")
        out = ROOT / "tb" / f"sim_{tag}.vvp"
        cc = subprocess.run(["iverilog", "-g2005", "-o", str(out), f"-DNVEC={N_VEC}",
                             f"-DSHIFT={s}", f"-DK2={k2}", f"-DVINIT={v0}",
                             f'-DVEC="{vec.name}"', str(ROOT / "rtl" / f"{top}.v"),
                             str(ROOT / "tb" / tb)], capture_output=True, text=True)
        if cc.returncode != 0:
            raise RuntimeError(f"iverilog {tag}: {cc.stderr[-800:]}")
        rr = subprocess.run(["vvp", str(out)], cwd=str(ROOT / "tb"), capture_output=True, text=True)
        mres = re.search(r"RESULT (\d+) mismatches of (\d+) vectors, (\d+) alarm", rr.stdout)
        if not mres:
            raise RuntimeError(f"vvp {tag}: {rr.stdout[-800:]}")
        n_bad, n_v, n_al = map(int, mres.groups())
        ref_alarms = int(sum(1 for w in al if w))
        bittrue[tag] = {"mismatches": n_bad, "vectors": n_v, "alarm_cycles_rtl": n_al,
                        "alarm_cycles_ref": ref_alarms, "v_init": v0, "K2": k2,
                        "pass": n_bad == 0 and n_al == ref_alarms}
        log.info("Phase F %s: %d mismatches of %d vectors, alarms rtl %d / ref %d -> %s",
                 tag, n_bad, n_v, n_al, ref_alarms, "PASS" if bittrue[tag]["pass"] else "FAIL")
    io_utils.save_json(bittrue, paths.run_dir / "bittrue.json")

    # ---------------- synthesis ----------------
    area = {}
    for top, s, k2, kval in variants:
        tag = f"{top}_8_4_s{s}"
        ys = ROOT / "synth" / f"gen_{tag}.ys"
        kname = "K1" if top == "online_checker_l1" else "K2"
        params = f"-set BY {BY} -set LOGM {LOGM} -set S {s} -set KF {KF} -set {kname} {k2}"
        if top == "online_array":
            params += f" -set LOGK {LOGK}"
        ys.write_text("\n".join([
            f"read_verilog rtl/{top}.v",
            f"chparam {params} {top}",
            f"hierarchy -top {top}",
            f"synth -top {top}",
            f"dfflibmap -liberty {LIB}",
            f"abc -liberty {LIB}",
            "opt_clean",
            f"stat -liberty {LIB}",
            f"write_verilog -noattr synth/out/{tag}.v",
        ]) + "\n", encoding="utf-8")
        rr = subprocess.run(["yosys", "-q", "-s", str(ys.relative_to(ROOT)), "-l",
                             f"synth/out/{tag}.log"], cwd=str(ROOT), capture_output=True, text=True)
        txt = (ROOT / "synth" / "out" / f"{tag}.log").read_text(encoding="utf-8")
        cells = int(re.findall(r"Number of cells:\s+(\d+)", txt)[-1])
        a_um2 = float(re.findall(r"Chip area for module .*?: ([0-9.]+)", txt)[-1])
        groups = K if top == "online_array" else 1
        area[tag] = {"cells": cells, "area_um2": a_um2, "GE": a_um2 / GE_UM2,
                     "groups": groups, "GE_per_group": a_um2 / GE_UM2 / groups,
                     "GE_per_col": a_um2 / GE_UM2 / groups / M, "k": kval, "K2": k2, "S": s}
        log.info("Phase S %s: cells %d, area %.4f um2, %.1f GE, %.1f GE per 16-column group",
                 tag, cells, a_um2, a_um2 / GE_UM2, a_um2 / GE_UM2 / groups)
    io_utils.save_json(area, paths.run_dir / "area.json")
    log.info("E8.2 run %s complete in %.1f s; every number above traces to %s",
             paths.runstamp, time.perf_counter() - t_start, paths.run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
