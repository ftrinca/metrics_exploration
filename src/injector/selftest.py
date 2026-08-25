from __future__ import annotations

import sys

import numpy as np

from injector import distortions as D
from injector.reactivity.calibrate import calibration_table, solve_series
from injector.config import (DAMAGE_LEVELS, DISTORTIONS, DISTORTION_NAMES, SEED,
                             RESPONSE_RATE, TARGET_DAMAGE)

T, N = 1000, 6


def synthetic(seed=0):
    """Return z-scored series with both periodicity and autocorrelation."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    out = np.zeros((T, N))
    for i in range(N):
        season = np.sin(2 * np.pi * t / (40 + 7 * i)) + 0.5 * np.sin(2 * np.pi * t / 11)
        walk = np.cumsum(rng.normal(0, 0.06, T))
        noise = rng.normal(0, 0.35, T)
        x = season + walk + noise
        out[:, i] = (x - x.mean()) / x.std()
    return out


def mcar_mask(y, rate=0.4, seed=2):
    """Mask individual positions at random, independently in every series."""
    rng = np.random.default_rng(seed)
    return rng.random(y.shape) < rate


def scattered_mask(y, rate=0.4, seed=1):
    """Mask one contiguous block per series, each at its own random start."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(y.shape, dtype=bool)
    length = int(rate * y.shape[0])
    for i in range(y.shape[1]):
        start = rng.integers(0, y.shape[0] - length)
        mask[start:start + length, i] = True
    return mask


def blackout_mask(y, rate=0.4, seed=3):
    """Mask one contiguous block at the same position in every series."""
    rng = np.random.default_rng(seed)
    length = int(rate * y.shape[0])
    start = int(rng.integers(0, y.shape[0] - length))
    mask = np.zeros(y.shape, dtype=bool)
    mask[start:start + length, :] = True
    return mask


def check_calibration(y, mask, label, target=TARGET_DAMAGE):
    """Check that every distortion can be solved to the target, and report the misses."""
    print(f"\n### calibration — {label}, target {target}")
    calib, failures = {}, []
    for name in DISTORTION_NAMES:
        sev, ach, ok = {}, {}, {}
        for i in range(y.shape[1]):
            idx = np.where(mask[:, i])[0]
            if idx.size == 0:
                continue
            s, a, r = solve_series(name, y[:, i], idx, SEED + 1000 + i, target)
            sev[i], ach[i], ok[i] = s, a, r
        calib[name] = {"severity": sev, "achieved": ach, "reached": ok}
        if not all(ok.values()):
            missed = [i for i, r in ok.items() if not r]
            worst = max(abs(ach[i] - target) for i in missed)
            failures.append((name, len(missed), worst))
    print(calibration_table(calib, target))
    if failures:
        print("\n  NOT REACHED:")
        for name, n, worst in failures:
            print(f"    {name}: {n} series off target, worst gap {worst:.4f} sigma")
    return calib, failures


def check_structural(y, mask):
    """Check the declared invariants on the arrays themselves, returning the number of failures."""
    print("\n### structural invariants (array level, no metrics involved)")
    rows = []
    for name in DISTORTION_NAMES:
        preserves = DISTORTIONS[name]["preserves"]
        for i in range(y.shape[1]):
            idx = np.where(mask[:, i])[0]
            y_col = y[:, i]
            sev, _, _ = solve_series(name, y_col, idx, SEED + 1000 + i, TARGET_DAMAGE)
            vals = D.FUNCTIONS[name](y_col, idx, sev, SEED + 1000 + i)
            truth = y_col[idx]

            if "multiset" in preserves:
                ok = np.allclose(np.sort(vals), np.sort(truth), atol=1e-12)
                rows.append((name, "multiset", i, ok))
            if "mean" in preserves:
                ok = abs(float(vals.mean()) - float(truth.mean())) < 1e-9
                rows.append((name, "mean", i, ok))
            if "affine" in preserves:
                c = float(np.corrcoef(truth, vals)[0, 1])
                rows.append((name, "affine", i, abs(c - 1.0) < 1e-9))
            if "rank" in preserves:
                order = np.argsort(truth)
                ok = np.all(np.diff(vals[order]) >= -1e-12)
                rows.append((name, "rank", i, ok))

    by_claim = {}
    for name, claim, _, ok in rows:
        key = (name, claim)
        by_claim.setdefault(key, []).append(ok)
    failed = 0
    for (name, claim), oks in sorted(by_claim.items()):
        status = "ok" if all(oks) else f"FAIL ({oks.count(False)}/{len(oks)} series)"
        if not all(oks):
            failed += 1
        print(f"  {name:<12}{claim:<10}{status}")
    return failed


def check_response_reachable(y, mask):
    """Check that every distortion can reach every level of the damage-response curve."""
    print("\n### damage-response levels reachable")
    bad = []
    for target in DAMAGE_LEVELS:
        for name in DISTORTION_NAMES:
            for i in range(y.shape[1]):
                idx = np.where(mask[:, i])[0]
                _, got, ok = solve_series(name, y[:, i], idx, SEED + 1000 + i, target)
                if not ok:
                    bad.append((name, target, i, got))
    if not bad:
        print(f"  all {len(DAMAGE_LEVELS)} levels reachable by all "
              f"{len(DISTORTION_NAMES)} distortions")
    else:
        worst = {}
        for name, target, i, got in bad:
            worst.setdefault(name, []).append((target, got))
        for name, items in worst.items():
            tgts = sorted({t for t, _ in items})
            print(f"  {name}: unreachable at damage {tgts}")
    return bad


def main():
    """Run every check on synthetic data. Returns 0 when all of them pass."""
    print("=" * 72)
    print("INJECTOR SELF-TEST  (synthetic data, no ImputeGAP required)")
    print("=" * 72)
    y = synthetic()
    print(f"series: {y.shape}, per-series mean {y.mean(0).round(3)}, std {y.std(0).round(3)}")

    fails = 0
    for label, mask in [("mcar", mcar_mask(y)),
                        ("scattered", scattered_mask(y)),
                        ("blackout", blackout_mask(y))]:
        _, cal_fail = check_calibration(y, mask, label)
        fails += len(cal_fail)

    fails += check_structural(y, scattered_mask(y))
    fails += len(check_response_reachable(y, blackout_mask(y, rate=RESPONSE_RATE)))

    print("\n" + "=" * 72)
    print("SELF-TEST PASSED" if fails == 0 else f"SELF-TEST: {fails} problem(s) above")
    print("=" * 72)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
