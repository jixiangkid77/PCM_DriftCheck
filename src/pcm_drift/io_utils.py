"""Run directories, JST logging, config snapshots.

Traceability discipline (spec section 1): every number in a paper or report
must point back to one runstamp directory and its log. Runstamp directories are
never reused; existing files are never overwritten.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import platform
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

__all__ = [
    "JST",
    "jst_now",
    "make_run_dir",
    "resolve_paths",
    "get_logger",
    "save_json",
    "config_snapshot",
    "Paths",
]

JST = timezone(timedelta(hours=9), "JST")


def jst_now() -> datetime:
    return datetime.now(JST)


def make_run_dir(out_root: Path, suffix: str = ""):
    """Create <out_root>/<runstamp>/ with runstamp = YYYYMMDD_HHMM (JST).

    suffix (e.g. '_smoke') is appended to the runstamp so verification runs are
    distinguishable from data runs at a glance. On collision (two runs within
    one minute) fall back to YYYYMMDD_HHMMSS, then numeric suffixes. Never
    reuses an existing directory.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = jst_now().strftime("%Y%m%d_%H%M") + suffix
    cand = out_root / stamp
    if cand.exists():
        stamp = jst_now().strftime("%Y%m%d_%H%M%S") + suffix
        cand = out_root / stamp
        i = 2
        while cand.exists():
            cand = out_root / f"{stamp}_{i}"
            i += 1
        stamp = cand.name
    cand.mkdir()
    return cand, stamp


@dataclasses.dataclass
class Paths:
    root: Path
    out_root: Path
    run_dir: Path
    runstamp: str
    logs: Path
    figures: Path


def resolve_paths(out: str, exp: str = "E1", suffix: str = "") -> Paths:
    """Derive the project-relative logs/ and figures/<exp>/ locations from --out.

    When out matches <root>/results/<exp>, siblings are <root>/logs and
    <root>/figures/<exp> per the frozen directory layout (spec section 1);
    otherwise logs/ and figures/<exp>/ are created next to out.
    """
    out_root = Path(out).resolve()
    if out_root.parent.name.lower() == "results":
        root = out_root.parent.parent
    else:
        root = out_root.parent
    run_dir, stamp = make_run_dir(out_root, suffix=suffix)
    logs = root / "logs"
    figures = root / "figures" / exp
    logs.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return Paths(root=root, out_root=out_root, run_dir=run_dir, runstamp=stamp, logs=logs, figures=figures)


class _JSTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, JST)
        return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S JST")


def get_logger(name: str, log_path: Path) -> logging.Logger:
    """File + console logger with JST-stamped lines."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = _JSTFormatter("[%(asctime)s] %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _json_default(o):
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def save_json(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def config_snapshot(cfg, args_dict: dict, extra: dict | None = None) -> dict:
    """Full parameter snapshot for config.json (traceability)."""
    import numpy as np

    snap = {
        "config": dataclasses.asdict(cfg),
        "cli_args": args_dict,
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "generated_jst": jst_now().strftime("%Y-%m-%d %H:%M:%S JST"),
    }
    if extra:
        snap.update(extra)
    return snap
