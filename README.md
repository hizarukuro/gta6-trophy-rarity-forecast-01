# GTA VI Trophy Rarity Forecaster

[![make.fun bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hizarukuro/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hizarukuro/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
Built with make.fun — referral code c23ct4sn

`main.py` forecasts, for every PlayStation 5 trophy of Grand Theft Auto
VI, the earn rate Exophase will print for it at a case's target instant
— the percent of Exophase's tracked players who have earned it — as
five quantiles (5%, 25%, 50%, 75%, 95%). It is scored by mean pinball
loss on the logit scale, so all of its modelling happens on the logit
scale. Python 3.13, standard library only, no network, deterministic:
the same case always produces the same forecast, byte for byte.

## The model

A rate is a ratio, earners / players tracked, and the two sides move
for different reasons. Launch night proved the point: at Exophase's
first refresh the tracked base jumped 638 → 3,868 and every rate fell,
not because anyone un-earned a trophy but because the denominator
moved. So the program never extrapolates a rate directly. It splits
each trophy's rate into its two counts — earners and non-earners —
and projects both.

Time runs on a damped clock, u = ln(1 + hours since release), which
stretches the frantic first hours and compresses later ones. The page
only changes when Exophase refreshes it, so consecutive duplicate
reads are collapsed into distinct snapshots; over the recent window
of snapshots, ln earners and ln non-earners are each fit by least
squares against u. Both projections are anchored on the last observed
counts (a fit's intercept error never jumps the forecast), slopes are
clamped to sane ranges, a trophy with few snapshots of its own has
its slopes shrunk toward the pooled slopes fit to the page's own
trophies-earned total and its complement, and both moves are damped
by span / (span + Δu): a trend earns trust in proportion to the
history behind it relative to the leap ahead. The median is the ratio
of the two projected counts. Projecting both sides is what keeps a
saturating trophy honest — its earners can only grow while its
non-earners shrink, so the forecast bends toward its ceiling instead
of blowing through it — and the median's logit move from the last
observed rate is capped as a final guard.

The spread is the median shifted by z·σ on the logit scale. The base
σ grows like a random walk, 0.115·√(horizon/90 min) — 0.115 at the
practice cases' 90 minutes, about 0.98 at the final's 6,510 minutes,
close to the house baseline's published 0.125 and 1.06 — and each
trophy adds its own terms in quadrature: the fit residuals of its
earners, non-earners and the players series, scaled by the
extrapolation distance; a counting term for trophies with few
earners; a rounding term for rates so small (or so near 100) that the
printed 0.01 steps are relatively large on the logit scale. A history
with almost no distinct snapshots proves nothing about the page's
dynamics, so it widens σ further, and σ is floored and capped so no
trophy is forecast with silly confidence or none. Edge cases are
handled throughout: rates of 0.00 and 100.00 (clamped the way the
scoring clamps them), trophies missing from a read, and a case with a
single read.

## Running it

The grader runs the program once per case with only that case's file
in the input directory:

    MAKEFUN_INPUT_DIR=<dir with *.in cases> \
    MAKEFUN_OUTPUT_DIR=<dir for *.out forecasts> \
    python3 main.py

With the environment variables unset it reads `/input` and writes
`/output`. Every `*.in` file found is processed in name order, and
each produces one `<id>.out`: a JSON object `{"trophies": [{"id": n,
"q": [q05, q25, q50, q75, q95]}, ...]}` with one row per trophy.
