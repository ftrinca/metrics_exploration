"""Solve each distortion's severity so that all eight cause the same damage.

Every distortion is solved per series to a target mean absolute error at the
masked positions, expressed in units of that series' own sigma. Targets that
cannot be reached are reported rather than silently clipped, because a target
set above one distortion's ceiling would quietly become "as damaged as this
distortion can be" for some of the eight and not for others, which is the
confound the equalisation exists to remove.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np

from injector import distortions as D
from injector.config import (
    DAMAGE_TOLERANCE, DISTORTION_NAMES, PATTERNS, RATES, SEED, TARGET_DAMAGE,
    rate_dir,
)

MAX_BISECT_STEPS = 60
MAX_EXPAND_STEPS = 24


def _damage_at(name, y_col, idx, severity, seed):
    vals = D.FUNCTIONS[name](y_col, idx, severity, seed)
    return D.damage(y_col, idx, vals)


def solve_series(name, y_col, idx, seed, target=TARGET_DAMAGE):
    """Solve one distortion's severity for one series.

    Returns (severity, achieved_damage, reached), where reached is True when
    the achieved damage is within config.DAMAGE_TOLERANCE of the target.
    """
    spec = D.SEVERITY_SPEC[name]

    if spec["kind"] == "scan":
        # Damage is continuous in the parameter but not monotone, so bisection
        # alone would land on whichever root it happened to bracket. Walk the
        # scan points, take the first interval that straddles the target, and
        # bisect inside it. With nothing straddling, fall back to the closest
        # point visited and report it as not reached rather than pretending.
        pts = list(spec["scan"](y_col, idx))
        prev_x = prev_d = None
        best = None
        for x in pts:
            d = _damage_at(name, y_col, idx, x, seed)
            err = abs(d - target)
            if best is None or err < best[0]:
                best = (err, float(x), d)
            if prev_d is not None and (prev_d - target) * (d - target) <= 0.0:
                lo_x, hi_x = prev_x, x
                for _ in range(MAX_BISECT_STEPS):
                    mid = 0.5 * (lo_x + hi_x)
                    d_mid = _damage_at(name, y_col, idx, mid, seed)
                    if abs(d_mid - target) <= DAMAGE_TOLERANCE:
                        return mid, d_mid, True
                    d_lo = _damage_at(name, y_col, idx, lo_x, seed)
                    if (d_lo - target) * (d_mid - target) <= 0.0:
                        hi_x = mid
                    else:
                        lo_x = mid
                mid = 0.5 * (lo_x + hi_x)
                d_mid = _damage_at(name, y_col, idx, mid, seed)
                return mid, d_mid, abs(d_mid - target) <= DAMAGE_TOLERANCE
            prev_x, prev_d = x, d
        err, sev, got = best
        return sev, got, err <= DAMAGE_TOLERANCE

    lo, hi = spec["lo"], spec["hi"]
    d_hi = _damage_at(name, y_col, idx, hi, seed)

    # Expand the upper bracket when the distortion has more headroom than the
    # default cap suggests. A bounded distortion (reorder is capped at a full
    # permutation, smoothing at the series mean) simply stops rising, so the
    # loop gives up rather than running away.
    steps = 0
    while d_hi < target and steps < MAX_EXPAND_STEPS:
        prev = d_hi
        hi *= 2.0
        d_hi = _damage_at(name, y_col, idx, hi, seed)
        steps += 1
        if d_hi <= prev + 1e-12:       # saturated: more severity buys nothing
            break

    if d_hi < target:                   # ceiling below the target
        return hi, d_hi, False

    for _ in range(MAX_BISECT_STEPS):
        mid = 0.5 * (lo + hi)
        d_mid = _damage_at(name, y_col, idx, mid, seed)
        if abs(d_mid - target) <= DAMAGE_TOLERANCE:
            return mid, d_mid, True
        if d_mid < target:
            lo = mid
        else:
            hi = mid

    mid = 0.5 * (lo + hi)
    d_mid = _damage_at(name, y_col, idx, mid, seed)
    return mid, d_mid, abs(d_mid - target) <= DAMAGE_TOLERANCE


def solve_scenario(y_true, mask, seed=SEED, target=TARGET_DAMAGE, names=None):
    """Solve every distortion for every series of one (pattern, rate).

    Returns {distortion: {"severity": {series_idx: value},
                          "achieved": {series_idx: damage},
                          "reached":  {series_idx: bool}}}
    with integer series indices, which become strings once written to JSON.
    Series whose missing block is flat are absent from all three maps.
    """
    names = names or DISTORTION_NAMES
    out = {}
    for name in names:
        sev, ach, ok = {}, {}, {}
        for series_idx in range(y_true.shape[1]):
            idx = np.where(mask[:, series_idx])[0]
            if idx.size == 0:
                continue
            y_col = y_true[:, series_idx]
            if D.gap_sigma(y_col, idx) <= 0.0:
                # A flat block has no sigma to scale against, so every severity
                # is meaningless here.
                continue
            s, a, r = solve_series(name, y_col, idx, seed + 1000 + series_idx, target)
            sev[series_idx], ach[series_idx], ok[series_idx] = float(s), float(a), bool(r)
        out[name] = {"severity": sev, "achieved": ach, "reached": ok}
    return out


def calibration_table(calib, target=TARGET_DAMAGE):
    """Render one line per distortion: mean severity, mean achieved damage,
    spread, and how many series reached the target."""
    lines = [
        f"CALIBRATION  (target damage = {target:.3f} sigma, "
        f"tolerance = {DAMAGE_TOLERANCE})",
        "=" * 78,
        f"{'distortion':<13}{'severity':>12}{'achieved':>11}{'spread':>9}{'reached':>10}",
        "-" * 78,
    ]
    for name, rec in calib.items():
        sev = np.array(list(rec["severity"].values()), dtype=float)
        ach = np.array(list(rec["achieved"].values()), dtype=float)
        ok = list(rec["reached"].values())
        if sev.size == 0:
            lines.append(f"{name:<13}{'-':>12}{'-':>11}{'-':>9}{'0/0':>10}")
            continue
        lines.append(
            f"{name:<13}{sev.mean():>12.4f}{ach.mean():>11.4f}"
            f"{ach.max() - ach.min():>9.4f}{f'{sum(ok)}/{len(ok)}':>10}"
        )
    lines.append("-" * 78)
    lines.append(
        "severity is in each distortion's own parameter (see config.DISTORTIONS); "
        "achieved and spread are in sigma units."
    )
    return "\n".join(lines)


def calibrate_phase(patterns, rates, target=TARGET_DAMAGE, force=False):
    from injector.build import load_ground_truth
    from core.missingness_patterns import make_mask

    print(f"Loading ground truth")
    y_true = load_ground_truth()
    print(f"  shape={y_true.shape}\n")

    for pattern in patterns:
        print(f"=== pattern: {pattern} " + "=" * 46)
        for rate in rates:
            path = os.path.join(rate_dir(pattern, rate), "calibration.json")
            if not force and os.path.exists(path):
                print(f"  -- rate {rate:.0%}: SKIP (already solved)")
                continue
            mask = make_mask(y_true, pattern, rate)
            calib = solve_scenario(y_true, mask, target=target)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump({"target": target, "distortions": calib}, f, indent=2)
            print(f"  -- rate {rate:.0%} --")
            print("\n".join("     " + l for l in calibration_table(calib, target).splitlines()))
            print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Injector v2 — calibrate severities to equal damage.")
    ap.add_argument("--patterns", nargs="+", default=PATTERNS, choices=PATTERNS)
    ap.add_argument("--rates", nargs="+", type=float, default=RATES, choices=RATES)
    ap.add_argument("--target", type=float, default=TARGET_DAMAGE)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    calibrate_phase(a.patterns, a.rates, target=a.target, force=a.force)
