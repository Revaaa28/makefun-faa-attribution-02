# polymarket-faa-attribution-02 entry (Python 3.13, standard library only).
#
# Method: persistence with data-driven bands. Prices on Polymarket are sticky and move in
# discrete ticks with occasional jumps, so a normal random-walk band (the house baseline)
# is the wrong shape: too wide in the middle, too thin in the tails. For each market we
# estimate the distribution of future price changes from the case's own recent history
# (the last 1440 minutes) in two complementary ways and blend them:
#
#   1. Scaled-shape component (weight 0.75): take the market's 2-minute price changes,
#      standardize them by their root-mean-square, and use their empirical quantiles as a
#      fixed shape that is then scaled by sigma * sqrt(h / 2) for horizon h. This captures
#      the fat-tailed, zero-heavy shape of real Polymarket moves.
#   2. Direct empirical component (weight 0.25): for each horizon h, the empirical
#      quantiles of the market's actual h-minute changes in the window. This captures
#      horizon effects the sqrt scaling misses (stickiness, mean reversion, bounds).
#
# The blended band is then narrowed by a factor 0.9 around its median (the raw estimates
# are slightly overdispersed), clamped to [0, 1], and made non-decreasing. The median
# stays at the last price plus the small empirical drift of the change distribution.
# On the kit's six practice cases this scores a mean pinball loss of about 0.289,
# against the house baseline's published 0.363589.
#
# I/O mirrors the kit starter exactly: read every <id>.in in the input directory
# (MAKEFUN_INPUT_DIR, default /input), write <id>.out to the output directory
# (MAKEFUN_OUTPUT_DIR, default /output) as {"forecasts": [{"id", "q"}]}.

import json
import math
import os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]
WINDOW = 1440          # minutes of history used for estimation
STEP = 2               # step (minutes) for the scaled-shape component
BLEND_EMPIRICAL = 0.25  # weight of the direct empirical component
WIDTH = 0.9            # band narrowing factor around the median
MIN_SAMPLES = 40       # minimum h-step samples for a direct empirical quantile


def quantile(sorted_vals, tau):
    """Linear-interpolation quantile of an ascending-sorted sequence."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def price6(x):
    """Clamp to [0, 1] and round to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)), 6) + 0.0


def scaled_shape_rows(seg, horizon):
    """Component 1: empirical shape of STEP-minute changes, scaled by sqrt(h / STEP)."""
    diffs = [seg[i + STEP] - seg[i] for i in range(len(seg) - STEP)]
    step = STEP
    if len(diffs) < 20:
        diffs = [seg[i + 1] - seg[i] for i in range(len(seg) - 1)]
        step = 1
    if not diffs:
        sigma = 0.0025
        shape = Z
        step = 1
    else:
        sigma = math.sqrt(sum(d * d for d in diffs) / len(diffs))
        if sigma <= 0.0:
            sigma = 1e-5
        ordered = sorted(d / sigma for d in diffs)
        shape = [quantile(ordered, tau) for tau in QUANTILES]
    x = seg[-1]
    return [[x + shape[j] * sigma * math.sqrt(h / step) for j in range(5)]
            for h in range(1, horizon + 1)]


def empirical_rows(seg, horizon):
    """Component 2: empirical quantiles of the actual h-minute changes, per horizon."""
    n = len(seg)
    by_h = {}
    for h in range(1, horizon + 1):
        if n - h >= MIN_SAMPLES:
            ordered = sorted(seg[i + h] - seg[i] for i in range(n - h))
            by_h[h] = [quantile(ordered, tau) for tau in QUANTILES]
    x = seg[-1]
    if not by_h:
        diffs = [seg[i + 1] - seg[i] for i in range(len(seg) - 1)]
        sigma = math.sqrt(sum(d * d for d in diffs) / len(diffs)) if diffs else 0.0025
        if sigma <= 0.0:
            sigma = 1e-5
        return [[x + Z[j] * sigma * math.sqrt(h) for j in range(5)]
                for h in range(1, horizon + 1)]
    keys = sorted(by_h)
    rows = []
    for h in range(1, horizon + 1):
        if h in by_h:
            q = by_h[h]
        elif h < keys[0]:
            ref = keys[0]
            q = [v * math.sqrt(h / ref) for v in by_h[ref]]
        else:
            ref = keys[-1]
            q = [v * math.sqrt(h / ref) for v in by_h[ref]]
        rows.append([x + v for v in q])
    return rows


def tidy(rows):
    """Clamp each row to [0, 1] and make it non-decreasing."""
    out = []
    for row in rows:
        r = [min(1.0, max(0.0, v)) for v in row]
        for j in range(1, 5):
            if r[j] < r[j - 1]:
                r[j] = r[j - 1]
        out.append(r)
    return out


def forecast(prices, horizon):
    if not prices:
        return [[0.5] * 5 for _ in range(horizon)]
    if len(prices) < 3:
        x = prices[-1]
        return [[price6(x + z * 0.0025 * math.sqrt(h)) for z in Z]
                for h in range(1, horizon + 1)]
    seg = prices[-WINDOW:]
    a = tidy(scaled_shape_rows(seg, horizon))
    b = tidy(empirical_rows(seg, horizon))
    rows = []
    for h in range(horizon):
        blended = [(1.0 - BLEND_EMPIRICAL) * a[h][j] + BLEND_EMPIRICAL * b[h][j]
                   for j in range(5)]
        med = blended[2]
        row = [med + (v - med) * WIDTH for v in blended]
        row = [min(1.0, max(0.0, v)) for v in row]
        for j in range(1, 5):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append([price6(v) for v in row])
    return rows


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(p for p in Path(INPUT_DIR).iterdir() if p.name.endswith(".in"))
    for path in inputs:
        case_id = path.name[: -len(".in")]
        case = json.loads(path.read_text())
        if list(case["quantiles"]) != QUANTILES:
            raise ValueError(f"{path.name} asks for quantiles {case['quantiles']}")
        forecasts = [{"id": m["id"], "q": forecast(m["prices"], case["horizon_minutes"])}
                     for m in case["markets"]]
        (out_dir / f"{case_id}.out").write_text(json.dumps({"forecasts": forecasts}) + "\n")


if __name__ == "__main__":
    main()
