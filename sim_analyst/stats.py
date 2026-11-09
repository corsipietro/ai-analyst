"""Pure numerical helpers used by the tools: no database, no model, easy to test.

Each function takes plain Python lists and returns plain Python values.
"""
from __future__ import annotations

import datetime
import decimal
import math
import uuid

import numpy as np
from scipy import stats as scipy_stats


KEY_DIGITS = 12   # significant digits for data values used as keys (groups, values of x, filters): only float noise is removed


def clean(value, digits: int = 3):
    """Make one value fit in JSON, and round floats to `digits` significant digits (integer-valued floats become int).

    This is the one place where values coming from the database are made JSON-safe, because every
    tool result goes through it: dates, datetimes and times become ISO strings, a timedelta a
    string, a Decimal a float, a UUID a string and bytes a hex string. Rounding never cuts the
    integer part (977.969 -> 978, 1.23456 -> 1.23, 1e-06 stays 1e-06).
    """
    if isinstance(value, (datetime.date, datetime.time)):        # datetime is a date
        return value.isoformat()
    if isinstance(value, (datetime.timedelta, uuid.UUID)):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, decimal.Decimal):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        if value != 0:
            value = round(value, max(0, digits - 1 - math.floor(math.log10(abs(value)))))
        if value.is_integer() and abs(value) < 1e15:
            return int(value)
        return value
    return value


def round_significant(value, digits: int = 12):
    """Round a float to `digits` significant digits, to remove floating point noise (400.00000000000006 -> 400.0).

    Integers are returned unchanged.
    """
    if isinstance(value, int):
        return value
    return float(f"{value:.{digits}g}")


def clean_row(row, n_keys: int = 0) -> list:
    """Apply clean() to every value of a row: the first n_keys values are data (KEY_DIGITS), the others measures."""
    return [clean(v, KEY_DIGITS if i < n_keys else 3) for i, v in enumerate(row)]


def nice_step(span: float, target_points: int = 12) -> float:
    """Return a round step (1, 2 or 5 times a power of ten) giving about target_points over span."""
    if span <= 0:
        return 1
    raw = span / target_points
    power = 10 ** math.floor(math.log10(raw))
    for factor in (1, 2, 5, 10):
        if raw <= factor * power:
            step = factor * power
            return int(step) if float(step).is_integer() else step
    return 10 * power


def linear_slope(xs: list[float], ys: list[float]) -> float | None:
    """Least-squares slope of ys against xs, or None with fewer than 2 distinct xs."""
    if len(xs) < 2 or len(set(xs)) < 2:
        return None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    x_centered = x - x.mean()
    return float((x_centered * (y - y.mean())).sum() / (x_centered ** 2).sum())


def sample_points(points: list[tuple], step: float, start: float, per_group: int | None = None,
                  tolerance: float = 1e-9) -> list[tuple]:
    """Thin a series to about one point per `step` of x, always keeping the last point.

    With `per_group` (the most points the normal sampling would give) given and at most that many points, all are kept (irregular x, such as
    20, 50, 80, 120, 200, 400, must not lose points). Otherwise the range is cut into intervals of width
    `step` starting at `start`, and the first point of every non-empty interval is kept. For x on the
    grid start + k*step this is the same as keeping the points exactly on the grid.
    """
    if per_group is not None and len(points) <= per_group:
        return list(points)
    kept, seen = [], set()
    for p in points:
        interval = math.floor((p[0] - start) / step + tolerance)
        if interval not in seen:
            seen.add(interval)
            kept.append(p)
    if points and kept[-1][0] != points[-1][0]:
        kept.append(points[-1])
    return kept


def turning_points(xs: list[float], means: list[float]) -> list[dict]:
    """Return the values of x where the series stops rising and starts falling (peak) or the reverse (trough).

    Flat steps (equal consecutive values) are skipped when deciding the direction, so a flat
    top or bottom is reported at its first x.
    """
    turns = []
    previous_sign, previous_index = 0, 0
    for i in range(1, len(means)):
        difference = means[i] - means[i - 1]
        sign = (difference > 0) - (difference < 0)
        if sign == 0:
            continue
        if previous_sign and sign != previous_sign:
            kind = "peak" if previous_sign > 0 else "trough"
            turns.append({"kind": kind, "x": xs[previous_index], "mean": means[previous_index]})
        previous_sign, previous_index = sign, i
    return turns


def shape_of(means: list[float]) -> str:
    """Describe a series as 'increasing', 'decreasing', 'constant' or 'not monotonic' (strictly, no tolerance)."""
    differences = [b - a for a, b in zip(means, means[1:])]
    if all(d == 0 for d in differences):
        return "constant"
    if all(d >= 0 for d in differences):
        return "increasing"
    if all(d <= 0 for d in differences):
        return "decreasing"
    return "not monotonic"


def trend_summary(points: list[tuple], step: float, last_fraction: float = 0.2,
                  sampled: list[tuple] | None = None, per_group: int | None = None) -> dict:
    """Summarize one group's series of (x, n, mean, min, max), sorted by x.

    Returns initial and final values, maximum and minimum of the mean with their x,
    the slope of the mean over the last part of the range of x (least squares),
    the shape and the turning points of the series sampled at `step`, and the sampled series (x and mean).
    `sampled` replaces the sampling at `step` when the points are not regularly spaced (months); with
    `per_group`, a series of at most that many points is not sampled (see sample_points).
    Shape and turning points are computed on the means rounded to 3 significant digits (the numbers the
    model sees), so a difference of 1e-12 is not a peak.
    """
    xs = [p[0] for p in points]
    means = [p[2] for p in points]
    start, end = xs[0], xs[-1]
    i_max = max(range(len(means)), key=means.__getitem__)
    i_min = min(range(len(means)), key=means.__getitem__)
    cut = end - last_fraction * (end - start)
    tail = [(t, m) for t, m in zip(xs, means) if t >= cut]
    if len(tail) < 3:
        tail = list(zip(xs, means))[-3:]
    sampled = sample_points(points, step, start, per_group) if sampled is None else sampled
    s_xs = [p[0] for p in sampled]
    s_means = [clean(p[2]) for p in sampled]
    first, last = points[0], points[-1]
    return {
        "observations": sorted({p[1] for p in points}),
        "initial": {"x": first[0], "mean": first[2]},
        "final": {"x": last[0], "mean": last[2], "min": last[3], "max": last[4],
                  "spread": last[4] - last[3]},
        "max_of_mean": {"x": xs[i_max], "mean": means[i_max]},
        "min_of_mean": {"x": xs[i_min], "mean": means[i_min]},
        "slope_last_part": {"from_x": tail[0][0], "to_x": tail[-1][0],
                            "slope_per_unit_of_x": linear_slope([t for t, _ in tail], [m for _, m in tail])},
        "shape_sampled": shape_of(s_means),
        "turning_points_sampled": turning_points(s_xs, s_means),
        "series_sampled": {"columns": ["x", "mean"], "rows": [[p[0], p[2]] for p in sampled]},
    }


def bootstrap_difference(a: list[float], b: list[float], resamples: int, seed: int,
                         confidence: float) -> dict:
    """Difference of means mean(b) - mean(a) with a percentile bootstrap interval.

    Each group is resampled with replacement on its own; the seed is fixed, so the
    same data always give the same interval.
    """
    rng = np.random.default_rng(seed)
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a_means = rng.choice(a_arr, size=(resamples, a_arr.size), replace=True).mean(axis=1)
    b_means = rng.choice(b_arr, size=(resamples, b_arr.size), replace=True).mean(axis=1)
    differences = b_means - a_means
    tail = (1 - confidence) / 2
    low, high = np.quantile(differences, [tail, 1 - tail])
    return {"difference": float(b_arr.mean() - a_arr.mean()),
            "interval": [float(low), float(high)]}


def welch_interval(a: list[float], b: list[float], confidence: float) -> dict:
    """Difference of means mean(b) - mean(a) with Welch's t interval (unequal variances).

    The standard error is sqrt(s_a^2/n_a + s_b^2/n_b); the degrees of freedom come from the
    Welch-Satterthwaite formula; the multiplier is the Student t quantile. Unlike the percentile
    bootstrap, it stays honest with few values per group, under the assumption that the values
    are roughly normal within each group. Needs at least 2 values per group.

    If both groups have no variability the standard error is 0 and no interval exists:
    interval and degrees_of_freedom are then None (the caller must say so).
    """
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    difference = float(b_arr.mean() - a_arr.mean())
    var_a, var_b = a_arr.var(ddof=1) / a_arr.size, b_arr.var(ddof=1) / b_arr.size
    if var_a + var_b == 0:
        return {"difference": difference, "interval": None, "degrees_of_freedom": None}
    error = math.sqrt(var_a + var_b)
    dof = (var_a + var_b) ** 2 / (var_a ** 2 / (a_arr.size - 1) + var_b ** 2 / (b_arr.size - 1))
    multiplier = float(scipy_stats.t.ppf(1 - (1 - confidence) / 2, dof))
    return {"difference": difference, "interval": [difference - multiplier * error, difference + multiplier * error],
            "degrees_of_freedom": float(dof)}
