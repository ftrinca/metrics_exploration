from __future__ import annotations

import argparse
import json
import os

import numpy as np

from metric_eval.experiments.injector import distortions as D
from metric_eval.experiments.injector.config import (
    DAMAGE_TOLERANCE, DISTORTION_NAMES, PATTERNS, RATES, SEED, TARGET_DAMAGE,
    pass_filename, rate_dir,
)

MAX_BISECT_STEPS = 60
MAX_EXPAND_STEPS = 24


def _damage_at(name, y_col, idx, severity, seed, damage_fn=D.damage):
    """Damage caused by one distortion at one severity, measured by damage_fn."""
    vals = D.FUNCTIONS[name](y_col, idx, severity, seed)
    return damage_fn(y_col, idx, vals)


def solve_series(name, y_col, idx, seed, target=TARGET_DAMAGE, damage_fn=D.damage):
    """Solve one distortion's severity for one series.

    Returns (severity, achieved_damage, reached), where reached is True when the
    achieved damage is within config.DAMAGE_TOLERANCE of the target. A target
    above the distortion's ceiling is reported as not reached, never clipped.
    `damage_fn` is the calibration target, one of distortions.DAMAGE_FUNCTIONS.
    """
    spec = D.SEVERITY_SPEC[name]

    if spec["kind"] == "scan":
        pts = list(spec["scan"](y_col, idx))
        prev_x = prev_d = None
        best = None
        for x in pts:
            d = _damage_at(name, y_col, idx, x, seed, damage_fn)
            err = abs(d - target)
            if best is None or err < best[0]:
                best = (err, float(x), d)
            if prev_d is not None and (prev_d - target) * (d - target) <= 0.0:
                lo_x, hi_x, d_lo = prev_x, x, prev_d
                for _ in range(MAX_BISECT_STEPS):
                    mid = 0.5 * (lo_x + hi_x)
                    d_mid = _damage_at(name, y_col, idx, mid, seed, damage_fn)
                    if abs(d_mid - target) <= DAMAGE_TOLERANCE:
                        return mid, d_mid, True
                    if (d_lo - target) * (d_mid - target) <= 0.0:
                        hi_x = mid
                    else:
                        lo_x, d_lo = mid, d_mid
                mid = 0.5 * (lo_x + hi_x)
                d_mid = _damage_at(name, y_col, idx, mid, seed, damage_fn)
                return mid, d_mid, abs(d_mid - target) <= DAMAGE_TOLERANCE
            prev_x, prev_d = x, d
        err, sev, got = best
        return sev, got, err <= DAMAGE_TOLERANCE

    lo, hi = spec["lo"], spec["hi"]
    d_hi = _damage_at(name, y_col, idx, hi, seed, damage_fn)

    steps = 0
    while d_hi < target and steps < MAX_EXPAND_STEPS:
        prev = d_hi
        hi *= 2.0
        d_hi = _damage_at(name, y_col, idx, hi, seed, damage_fn)
        steps += 1
        if d_hi <= prev + 1e-12:
            break

    if d_hi < target:
        return hi, d_hi, False

    for _ in range(MAX_BISECT_STEPS):
        mid = 0.5 * (lo + hi)
        d_mid = _damage_at(name, y_col, idx, mid, seed, damage_fn)
        if abs(d_mid - target) <= DAMAGE_TOLERANCE:
            return mid, d_mid, True
        if d_mid < target:
            lo = mid
        else:
            hi = mid

    mid = 0.5 * (lo + hi)
    d_mid = _damage_at(name, y_col, idx, mid, seed, damage_fn)
    return mid, d_mid, abs(d_mid - target) <= DAMAGE_TOLERANCE


def solve_scenario(y_true, mask, seed=SEED, target=TARGET_DAMAGE, names=None,
                   damage_fn=D.damage):
    """Solve every distortion for every series of one (pattern, rate).
    Returns {distortion: {"severity": {series_idx: value}, "achieved": {series_idx: damage}, "reached":  {series_idx: bool}}}.
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
                continue
            s, a, r = solve_series(name, y_col, idx, seed + 1000 + series_idx,
                                   target, damage_fn)
            sev[series_idx], ach[series_idx], ok[series_idx] = float(s), float(a), bool(r)
        out[name] = {"severity": sev, "achieved": ach, "reached": ok}
    return out


def calibration_table(calib, target=TARGET_DAMAGE):
    """Render one line per distortion: mean severity, mean achieved damage, spread, reached count."""
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
    lines.append("severity: each distortion's own parameter.  achieved, spread: sigma.")
    return "\n".join(lines)


def calibrate_phase(patterns, rates, target=TARGET_DAMAGE, force=False,
                    damage_metric="mae"):
    """Solve and cache the severities for every (pattern, rate).

    `damage_metric` picks the calibration target and the cache filename, so
    the MAE and the RMSE pass live side by side in the same rate directory.
    """
    from metric_eval.experiments.injector.reactivity.build import load_ground_truth
    from metric_eval.core.missingness_patterns import make_mask

    print("Loading ground truth")
    y_true = load_ground_truth()
    print(f"  shape={y_true.shape}\n")

    for pattern in patterns:
        print(f"=== pattern: {pattern} " + "=" * 46)
        for rate in rates:
            path = os.path.join(rate_dir(pattern, rate),
                                pass_filename("calibration.json", damage_metric))
            if not force and os.path.exists(path):
                print(f"  -- rate {rate:.0%}: SKIP (already solved)")
                continue
            mask = make_mask(y_true, pattern, rate)
            calib = solve_scenario(y_true, mask, target=target,
                                   damage_fn=D.DAMAGE_FUNCTIONS[damage_metric])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump({"target": target, "damage_metric": damage_metric,
                           "distortions": calib}, f, indent=2)
            print(f"  -- rate {rate:.0%} --")
            print("\n".join("     " + l for l in calibration_table(calib, target).splitlines()))
            print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Damage reactivity — calibrate severities to a common damage.")
    ap.add_argument("--patterns", nargs="+", default=PATTERNS, choices=PATTERNS)
    ap.add_argument("--rates", nargs="+", type=float, default=RATES, choices=RATES)
    ap.add_argument("--target", type=float, default=TARGET_DAMAGE)
    ap.add_argument("--damage-metric", default="mae", choices=list(D.DAMAGE_FUNCTIONS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    calibrate_phase(a.patterns, a.rates, target=a.target, force=a.force,
                    damage_metric=a.damage_metric)
