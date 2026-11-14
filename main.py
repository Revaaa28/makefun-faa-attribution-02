# polymarket-faa-attribution-02 entry (Python 3.13, standard library only).
#
# Method: persistence with an empirical-shape band, ensembled over two history
# windows, with a mild level-proportional fade. Prices on Polymarket are sticky,
# move in discrete ticks and jump occasionally, and their dispersion grows much
# more slowly than a random walk's sqrt(h) (variance is strongly sub-diffusive:
# these prices are bounded and mean-reverting). So instead of a normal band:
#
#   * Shape: for each window W in {720, 1440} minutes of the case's own history,
#     take the market's 20-minute price changes, standardize them by their
#     root-mean-square, and use their empirical quantiles as a fixed shape of
#     the change distribution (fat-tailed and zero-heavy, like real moves).
#   * Scale: multiply the shape by the market's empirical RMS of h-minute
#     changes at each horizon h (its own variance term structure), rather than
#     by sigma * sqrt(h). This is what captures the sub-diffusive saturation.
#   * The raw band is narrowed by a factor 0.65 around its median, clamped to
#     [0, 1] and made non-decreasing; the two windows' bands are averaged.
#   * Fade: the whole band is shifted down by 0.04 * x * (h / 240), where x is
#     the last price. In the practice event, higher-priced markets systematically
#     displaced slightly downward over the next four hours; the shift is zero at
#     the checkpoint and reaches 4% of the price at the horizon's end.
#   * Flat windows: a window whose price never moved has zero empirical scale
#     at every horizon, so its band would collapse to a spike on the last
#     price. It falls back to the generic 0.0025 * sqrt(h) scale instead (the
#     same fallback used for very short histories), narrowed as usual. No
#     practice window is flat, so this changes no practice score; it guards
#     the final's thin markets, which can sit at one price for hours and then
#     jump on news.
#
# On the kit's six practice cases this scores a mean pinball loss of 0.272,
# against the house baseline's published 0.363589 and this entry's previous
# version's 0.289. (Leave-one-case-out validation of the design: 0.279.)
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
WINDOWS = (720, 1440)  # history windows ensembled, minutes
STEP = 20              # step (minutes) of the changes defining the shape
WIDTH = 0.65           # band narrowing factor around the median
DECAY = 0.04           # fade: shift down by DECAY * price * (h / horizon)
MIN_SAMPLES = 40       # minimum h-step samples for an empirical RMS at h


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


def rms(vals):
    return math.sqrt(sum(v * v for v in vals) / len(vals)) if vals else 0.0


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


def window_rows(seg, horizon):
    """Band for one history window: empirical shape x empirical RMS(h) curve."""
    m = len(seg)
    x = seg[-1]
    # shape from STEP-minute changes (1-minute changes if the window is short)
    if m - STEP >= 20:
        diffs = [seg[i + STEP] - seg[i] for i in range(m - STEP)]
        keff = STEP
    else:
        diffs = [seg[i + 1] - seg[i] for i in range(m - 1)]
        keff = 1
    sigma = rms(diffs)
    # A window that never moved carries no scale information of its own: the
    # standardized changes and the RMS curve below are all zero, which would
    # collapse the band to a spike on the last price. Such a market is quiet,
    # not certain -- fall back to the generic scale (the same 0.0025 * sqrt(h)
    # used for very short histories) so it still gets an honest band.
    flat = sigma == 0.0
    sref = sigma if sigma > 0.0 else 1e-5
    if flat or not diffs:
        shape = list(Z)
    else:
        ordered = sorted(d / sref for d in diffs)
        shape = [quantile(ordered, tau) for tau in QUANTILES]
    # empirical RMS of h-minute changes, per horizon
    curve = [None] * (horizon + 1)
    for h in range(1, horizon + 1):
        if m - h >= MIN_SAMPLES:
            curve[h] = rms([seg[i + h] - seg[i] for i in range(m - h)])
    avail = [h for h in range(1, horizon + 1) if curve[h] is not None]
    rows = []
    for h in range(1, horizon + 1):
        if flat:
            s = 0.0025 * math.sqrt(h)
        elif avail:
            if curve[h] is not None:
                s = curve[h]
            elif h < avail[0]:
                s = curve[avail[0]] * math.sqrt(h / avail[0])
            else:
                s = curve[avail[-1]] * math.sqrt(h / avail[-1])
        else:
            s = sref * math.sqrt(h / keff)
        rows.append([x + shape[j] * s for j in range(5)])
    rows = tidy(rows)
    # narrow the band around its own median
    out = []
    for row in rows:
        med = row[2]
        out.append([med + (v - med) * WIDTH for v in row])
    return tidy(out)


def forecast(prices, horizon):
    if not prices:
        return [[0.5] * 5 for _ in range(horizon)]
    x = prices[-1]
    if len(prices) < 3:
        return [[price6(x + z * 0.0025 * math.sqrt(h)) for z in Z]
                for h in range(1, horizon + 1)]
    acc = None
    for w in WINDOWS:
        rows = window_rows(prices[-w:], horizon)
        if acc is None:
            acc = [list(r) for r in rows]
        else:
            for i in range(horizon):
                for j in range(5):
                    acc[i][j] += rows[i][j]
    rows = []
    for i in range(horizon):
        h = i + 1
        shift = DECAY * x * (h / horizon)
        row = [min(1.0, max(0.0, acc[i][j] / len(WINDOWS) - shift)) for j in range(5)]
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
