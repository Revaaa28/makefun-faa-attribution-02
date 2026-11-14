# FAA outage attribution forecaster — make.fun `polymarket-faa-attribution-02`

Entry point: `main.py` (Python 3.13, standard library only). Run once per case; it reads
every `<id>.in` in the input directory (`MAKEFUN_INPUT_DIR`, default `/input`) and writes
`<id>.out` forecasts to the output directory (`MAKEFUN_OUTPUT_DIR`, default `/output`):
for each market, 240 rows of five quantiles (5%, 25%, 50%, 75%, 95%) of the Yes price at
each minute after the checkpoint.

## Method

Persistence with data-driven bands. Polymarket prices are sticky, move in discrete
ticks, and jump occasionally — so a normal random-walk band is the wrong shape: too wide
in the middle, too thin in the tails. For each market, the distribution of future price
changes is estimated from the case's own last 1440 minutes in two complementary ways,
then blended:

1. **Scaled-shape component (weight 0.75).** The market's 2-minute price changes are
   standardized by their root-mean-square; their empirical quantiles form a fixed shape
   that is scaled by `sigma * sqrt(h / 2)` for horizon `h`. This captures the fat-tailed,
   zero-heavy shape of real moves.
2. **Direct empirical component (weight 0.25).** For each horizon `h`, the empirical
   quantiles of the market's actual `h`-minute changes. This captures horizon effects
   that square-root scaling misses (stickiness, mean reversion, the [0, 1] bounds).

The blended band is narrowed by a factor of 0.9 around its median (the raw estimates
are slightly overdispersed), clamped to [0, 1], and made non-decreasing. The median sits
at the last price plus the small empirical drift of the change distribution.

## Validation

On the kit's six practice cases (Polymarket "Next CEO of NVIDIA?" markets), scored
against the true post-checkpoint prices from Polymarket's public price history on the
same 1-minute grid the kit uses, this scores a mean pinball loss of about **0.289**,
against the house baseline's published **0.363589**. (The practice event differs from
the final's, so the practice score is a guide, not a promise.)
