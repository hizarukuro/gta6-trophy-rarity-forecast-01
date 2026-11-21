#!/usr/bin/env python3
"""Forecast GTA VI PS5 trophy earn rates on Exophase at a future target.

Reads every *.in case file in $MAKEFUN_INPUT_DIR (else /input), in name
order, and writes one <id>.out forecast per case to $MAKEFUN_OUTPUT_DIR
(else /output). Each forecast gives, for every trophy in the case, five
quantiles of its earn rate (percent of tracked players who have earned
it) at the case's target instant. Scoring is mean pinball loss on the
logit scale, so all modelling happens on the logit scale.

Model
-----
A rate is a ratio: earners / players_tracked. The two series move for
different reasons -- the player base grows as adoption spreads and
decelerates, while a trophy's earners grow as the tracked base plays
toward it and saturates -- and launch showed the ratio can fall purely
because the denominator jumped (638 -> 3,868 players at the first
refresh, every rate down). So the program projects the two series
separately and takes the ratio at the target:

* time runs in u = ln(1 + hours since 2026-11-19T00:00:00Z), a clock
  that stretches early hours and compresses later ones, which damps
  every trend the way adoption and play-through curves damp;
* each trophy's rate is split into its two counts, earners and
  non-earners (players minus earners), and ln of each count is fit by
  least squares against u over the recent window of distinct page
  snapshots (the page only changes when Exophase refreshes it, so
  consecutive duplicate reads are collapsed first). Projecting both
  sides is what keeps a saturating trophy sane: its earners can grow
  only while its non-earners shrink, and the ratio of the two
  projections bends toward its ceiling on its own;
* projections are anchored on the last observed values, so a fit's
  intercept error never jumps the forecast, and both slopes' moves
  are damped by span / (span + delta_u) -- a trend earns trust in
  proportion to the history behind it relative to the leap ahead;
* a trophy's slopes are shrunk toward the pooled slopes (fit to the
  page's own trophies_earned total and its complement) when the
  trophy has few snapshots of its own;
* the median's logit move from the last observed rate is capped, so
  no trophy is extrapolated anywhere absurd.

Spread
------
Quantiles are the median shifted by z * sigma on the logit scale. The
base sigma grows like a random walk, 0.115 * sqrt(horizon_minutes/90)
(0.115 at 90 minutes, about 0.98 at the final's 6,510 minutes, close to
the house baseline's published 0.125 and 1.06), and each trophy adds
its own terms in quadrature: the fit residuals of its earners series
and of the players series scaled by the extrapolation distance, a
counting term for trophies with few earners, and a rounding term for
rates so small that the printed 0.01 steps are relatively large. A
history with almost no distinct snapshots proves nothing about the
page's dynamics, so it widens sigma further. Sigma is floored and
capped so no trophy is ever forecast with silly confidence or none.

Deterministic: no randomness, no clock, no network. Standard library
only.
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone
from statistics import NormalDist

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
RATE_LO = 0.01          # scoring clamps rates into [0.01, 99.99]
RATE_HI = 99.99
SIGMA_COEF = 0.115      # base sigma at a 90-minute horizon
SIGMA_MIN = 0.06
SIGMA_MAX = 2.75
MAX_LOGIT_MOVE = 3.0    # cap on the median's logit move from last rate
WINDOW_HOURS = 48.0     # fit window: the last two days of snapshots
WINDOW_MIN_SNAPS = 5    # ... when it still leaves this many snapshots
SHRINK_K = 5.0          # snapshots of own history vs pooled slope
PSEUDO_EARNERS = 0.5    # continuity correction for zero-count fits

_NORMAL = NormalDist()


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp into an aware datetime."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def time_u(moment):
    """The damped clock: ln(1 + hours since release), floored."""
    hours = (moment - RELEASE).total_seconds() / 3600.0
    return math.log(max(1.0 + hours, 0.05))


def clamp_rate(rate):
    return min(max(rate, RATE_LO), RATE_HI)


def logit_of_percent(rate):
    p = clamp_rate(rate) / 100.0
    return math.log(p / (1.0 - p))


def percent_of_logit(value):
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    exp_v = math.exp(value)
    return 100.0 * exp_v / (1.0 + exp_v)


def fit_line(xs, ys):
    """Least squares y ~ a + b*x; returns (slope, rms residual).

    The intercept is not returned: projections anchor on the last
    observed value and use only the slope, so a level shift in the
    history (a refresh that moved everything at once) does not drag
    the forecast with it.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return 0.0, 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:
        return 0.0, 0.0
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    rms = math.sqrt(sum((y - (intercept + slope * x)) ** 2
                        for x, y in zip(xs, ys)) / n)
    return slope, rms


def distinct_snapshots(history):
    """Collapse consecutive duplicate reads, keeping first-seen times.

    Exophase refreshes on its own schedule, so long runs of identical
    reads are one page state, not many observations. A snapshot is
    (moment, players_tracked, trophies_earned, rates tuple).
    """
    snaps = []
    for read in history:
        rates = tuple(read.get("rates") or ())
        key = (read.get("players_tracked"), rates)
        if snaps and snaps[-1][3] == key:
            continue
        moment = parse_time(read.get("slot") or read.get("at"))
        snaps.append((moment, read.get("players_tracked"),
                      read.get("trophies_earned"), key))
    return snaps


def window_snapshots(snaps):
    """The recent window for fitting, falling back to all snapshots."""
    if len(snaps) <= WINDOW_MIN_SNAPS:
        return snaps
    cutoff = snaps[-1][0] - timedelta(hours=WINDOW_HOURS)
    recent = [s for s in snaps if s[0] >= cutoff]
    if len(recent) >= WINDOW_MIN_SNAPS:
        return recent
    return snaps


def last_known(history, index):
    """Last (players, rate) with a non-null rate for trophy `index`."""
    for read in reversed(history):
        rates = read.get("rates") or ()
        if index < len(rates) and rates[index] is not None:
            return read.get("players_tracked"), rates[index]
    return None, None


def forecast_case(case):
    """Return the output rows for one case: [{id, q: [5 quantiles]}]."""
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon = float(case.get("horizon_minutes") or 0.0)
    checkpoint = parse_time(case["at"])
    if case.get("target"):
        target = parse_time(case["target"])
    else:
        target = checkpoint + timedelta(minutes=horizon)
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    zscores = [_NORMAL.inv_cdf(level) for level in levels]

    snaps = distinct_snapshots(history)
    fit_snaps = window_snapshots(snaps)
    u_last = time_u(snaps[-1][0]) if snaps else time_u(checkpoint)
    u_target = time_u(target)
    delta_u = max(u_target - u_last, 0.0)
    span_u = 0.0
    if len(fit_snaps) >= 2:
        span_u = time_u(fit_snaps[-1][0]) - time_u(fit_snaps[0][0])

    # The last read's player count: the anchor every per-trophy count
    # is measured against, and the fallback when a trophy's own anchor
    # read lacks one.
    players_last = None
    for read in reversed(history):
        if read.get("players_tracked"):
            players_last = float(read["players_tracked"])
            break
    if players_last is None:
        players_last = 1.0
    # The players fit itself is not projected any more (the per-trophy
    # count projections carry the player base with them); its residual
    # still measures how jumpy the page's denominator is, which feeds
    # every trophy's spread below.
    p_xs = [time_u(s[0]) for s in fit_snaps if s[1]]
    p_ys = [math.log(float(s[1])) for s in fit_snaps if s[1]]
    _, resid_p = fit_line(p_xs, p_ys)

    # Pooled earners and non-earners slopes from the page's own totals
    # (trophies_earned, and players minus it): the shrinkage targets
    # for trophies with thin histories of their own.
    e_xs = [time_u(s[0]) for s in fit_snaps if s[2] is not None]
    e_ys = [math.log(max(float(s[2]), PSEUDO_EARNERS))
            for s in fit_snaps if s[2] is not None]
    slope_pool, _ = fit_line(e_xs, e_ys)
    slope_pool = min(max(slope_pool, 0.0), 4.0)
    g_xs = [time_u(s[0]) for s in fit_snaps
            if s[1] and s[2] is not None]
    g_ys = [math.log(max(float(s[1]) - float(s[2]), PSEUDO_EARNERS))
            for s in fit_snaps if s[1] and s[2] is not None]
    slope_pool_g, _ = fit_line(g_xs, g_ys)
    slope_pool_g = min(max(slope_pool_g, -4.0), 4.0)
    # A trend earns trust in proportion to the history behind it
    # relative to the leap ahead of it.
    damp = span_u / (span_u + delta_u) if span_u > 0.0 else 0.0

    base_sigma = SIGMA_COEF * math.sqrt(max(horizon, 1.0) / 90.0)
    if len(snaps) <= 1:
        snap_factor = 1.8
    elif len(snaps) <= 3:
        snap_factor = 1.35
    else:
        snap_factor = 1.0
    extrap = 0.75 + delta_u / (span_u + 1.0)

    rows = []
    for index, trophy in enumerate(trophies):
        anchor_players, anchor_rate = last_known(history, index)
        if anchor_rate is None:
            # Never seen: the smallest printable rate, honestly wide.
            median = RATE_LO
            sigma = min(SIGMA_MAX, base_sigma * 1.8)
            rows.append(_row(trophy["id"], median, sigma, zscores))
            continue
        if not anchor_players:
            anchor_players = players_last
        earners_last = anchor_rate / 100.0 * float(anchor_players)

        # This trophy's earners and non-earners over the fit window.
        t_xs, e_ys, g_ys = [], [], []
        for snap in fit_snaps:
            rates = snap[3][1]
            if index < len(rates) and rates[index] is not None and snap[1]:
                players = float(snap[1])
                earners = rates[index] / 100.0 * players
                t_xs.append(time_u(snap[0]))
                e_ys.append(math.log(max(earners, PSEUDO_EARNERS)))
                g_ys.append(math.log(max(players - earners,
                                         PSEUDO_EARNERS)))
        slope_e, resid_e = fit_line(t_xs, e_ys)
        slope_e = min(max(slope_e, 0.0), 4.0)
        slope_g, resid_g = fit_line(t_xs, g_ys)
        slope_g = min(max(slope_g, -4.0), 4.0)
        own = len(t_xs)
        weight = own / (own + SHRINK_K) if own else 0.0
        slope_e = weight * slope_e + (1.0 - weight) * slope_pool
        slope_g = weight * slope_g + (1.0 - weight) * slope_pool_g

        # Project both counts from their anchors with damped moves and
        # take the ratio: the rate bends toward its ceiling because
        # the non-earners a hot trophy still needs are disappearing.
        earners_anchor = earners_last if earners_last > 0.0 \
            else PSEUDO_EARNERS
        gap_anchor = float(anchor_players) - earners_last
        if gap_anchor <= 0.0:
            gap_anchor = PSEUDO_EARNERS
        earners_target = earners_anchor * math.exp(
            slope_e * delta_u * damp)
        gap_target = gap_anchor * math.exp(slope_g * delta_u * damp)
        median = clamp_rate(
            100.0 * earners_target / (earners_target + gap_target))

        # Keep the median's logit within a sane move of the last rate.
        med_logit = logit_of_percent(median)
        last_logit = logit_of_percent(anchor_rate)
        if med_logit - last_logit > MAX_LOGIT_MOVE:
            median = percent_of_logit(last_logit + MAX_LOGIT_MOVE)
        elif last_logit - med_logit > MAX_LOGIT_MOVE:
            median = percent_of_logit(last_logit - MAX_LOGIT_MOVE)
        median = clamp_rate(median)

        fit_sigma = math.sqrt(resid_e ** 2 + resid_g ** 2
                              + resid_p ** 2) * extrap
        # Print rounding: rates carry a uniform +-0.005 error (sd 0.0029),
        # and d(logit)/d(rate in percent) is 1 / (100 * p * (1 - p)), so
        # the rounding noise on the logit scale grows as a rate nears
        # either clamp and is negligible in between.
        rate_frac = clamp_rate(anchor_rate) / 100.0
        round_sigma = min(0.30, 0.0029 / (100.0 * max(
            rate_frac * (1.0 - rate_frac), 1e-6)))
        count_sigma = min(0.50, 0.5 / math.sqrt(max(earners_last, 1.0)))
        sigma = math.sqrt(base_sigma ** 2 + fit_sigma ** 2
                          + round_sigma ** 2 + count_sigma ** 2)
        sigma = min(max(sigma * snap_factor, SIGMA_MIN), SIGMA_MAX)
        rows.append(_row(trophy["id"], median, sigma, zscores))
    return rows


def _row(trophy_id, median, sigma, zscores):
    med_logit = logit_of_percent(median)
    quantiles = []
    for z in zscores:
        q = percent_of_logit(med_logit + z * sigma)
        quantiles.append(round(clamp_rate(q), 4))
    for i in range(1, len(quantiles)):
        if quantiles[i] < quantiles[i - 1]:
            quantiles[i] = quantiles[i - 1]
    return {"id": trophy_id, "q": quantiles}


def fallback_rows(case):
    """Last-rate rows with the base spread, if anything above failed."""
    horizon = float(case.get("horizon_minutes") or 0.0)
    sigma = SIGMA_COEF * math.sqrt(max(horizon, 1.0) / 90.0)
    sigma = min(max(sigma * 1.5, SIGMA_MIN), SIGMA_MAX)
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    zscores = [_NORMAL.inv_cdf(level) for level in levels]
    history = case.get("history") or []
    rows = []
    for index, trophy in enumerate(case["trophies"]):
        _, rate = last_known(history, index)
        median = clamp_rate(rate) if rate is not None else RATE_LO
        rows.append(_row(trophy["id"], median, sigma, zscores))
    return rows


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(name for name in os.listdir(input_dir)
                   if name.endswith(".in"))
    for name in names:
        with open(os.path.join(input_dir, name), encoding="utf-8") as fh:
            case = json.load(fh)
        try:
            rows = forecast_case(case)
        except Exception:
            rows = fallback_rows(case)
        payload = json.dumps({"trophies": rows}, ensure_ascii=False,
                             separators=(",", ":"))
        out_name = str(case.get("id") or name[:-3]) + ".out"
        with open(os.path.join(output_dir, out_name), "w",
                  encoding="utf-8") as fh:
            fh.write(payload)


if __name__ == "__main__":
    main()
