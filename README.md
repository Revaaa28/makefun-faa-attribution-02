# FAA outage attribution forecaster — make.fun `polymarket-faa-attribution-02`

Entry point: `main.py` (Python 3.13, standard library only). Run once per case; it reads
every `<id>.in` in the input directory (`MAKEFUN_INPUT_DIR`, default `/input`) and writes
`<id>.out` forecasts to the output directory (`MAKEFUN_OUTPUT_DIR`, default `/output`):
for each market, 240 rows of five quantiles (5%, 25%, 50%, 75%, 95%) of the Yes price at
each minute after the checkpoint.

## Method

Persistence with data-driven bands. Polymarket prices are sticky, move in discrete
ticks, and jump occasionally — so a normal random-walk band is the wrong shape: too wide
in the middle, too thin in the tails. Their dispersion also grows far more slowly than
`sqrt(h)`: variance is strongly sub-diffusive because the prices are bounded and
mean-reverting. The forecaster therefore separates the *shape* of the change
distribution from its *scale at each horizon*:

1. **Shape.** For each of two history windows (the last 720 and the last 1440 minutes),
   the market's 20-minute price changes are standardized by their root-mean-square;
   their empirical quantiles form a fixed shape — fat-tailed and zero-heavy, like real
   Polymarket moves.
2. **Scale.** The shape is multiplied, horizon by horizon, by the market's *empirical*
   RMS of `h`-minute changes — its own variance term structure — rather than by
   `sigma * sqrt(h)`. This is what captures the sub-diffusive saturation of dispersion
   at long horizons.
3. **Calibration.** Each window's band is narrowed by a factor of 0.65 around its
   median, clamped to [0, 1] and made non-decreasing; the two windows' bands are
   averaged.
4. **Fade.** The averaged band is shifted down by `0.04 * price * (h / 240)`. In the
   practice event, higher-priced markets systematically displaced slightly downward
   over the next four hours; the shift is zero at the checkpoint and reaches 4% of the
   price at the horizon's end.

Directly forecasting with the empirical quantiles of `h`-minute changes, trend/momentum
extrapolation, mean reversion to the trailing mean, robust (IQR/MAD) scale estimators,
cross-market pooled shapes, and logit/arcsine re-parameterizations were all built and
measured; each scored worse than the design above, so none of them is in it.

## Validation

On the kit's six practice cases (Polymarket "Next CEO of NVIDIA?" markets), scored
against the true post-checkpoint prices from Polymarket's public price history on the
same 1-minute grid the kit uses, this scores a mean pinball loss of **0.272**, against
the house baseline's published **0.363589** and this entry's previous version's 0.289.
Design selection was validated leave-one-case-out (tune on five cases, test on the
sixth): the selection procedure scores **0.279**, so most of the improvement is
structural rather than practice-set fitting. (The practice event differs from the
final's, so the practice score is a guide, not a promise.)
