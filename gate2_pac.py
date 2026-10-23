"""PAC-h margin core, audited against Maxkoz777/overload-budget-autoscaling v1.2.0.

Upstream: Maxim Kozhinov and Manuel Mazzara, MIT license (see docs/gate2-pac.md).
The local minute-t command uses observations through t and forecasts truth t+h.
This module supplies requests; the shared actuator enforces physical constraints.
"""

from bisect import bisect_left, insort
from functools import lru_cache

import numpy as np
from scipy.stats import binom

from gate2_kernel import njit  # Reuse the project-local Numba dependency bootstrap.


@njit(cache=True)
def _margin_columns(scores, horizon, window, ranks, start):
    length = len(scores) + horizon
    ordered = np.sort(scores[start - horizon - window + 1:start - horizon + 1])
    paths = np.empty((len(ranks), length - start))
    for t in range(start, length):
        for row in range(len(ranks)):
            paths[row, t - start] = ordered[ranks[row] - 1]
        if t + 1 < length:
            old = scores[t - horizon - window + 1]
            pos = np.searchsorted(ordered, old)
            for j in range(pos, window - 1):
                ordered[j] = ordered[j + 1]
            new = scores[t - horizon + 1]
            pos = np.searchsorted(ordered[:window - 1], new)
            for j in range(window - 1, pos, -1):
                ordered[j] = ordered[j - 1]
            ordered[pos] = new
    return paths


@lru_cache(maxsize=None)
def pac_rank(window, delta, eta=.05):
    """Smallest 1-based l with P[Bin(window, 1-delta) >= l] <= eta.

    None means no finite observed-score order statistic satisfies the PAC rank.
    Binary search preserves the upstream inequality while avoiding W SciPy calls.
    """
    if isinstance(window, bool) or int(window) != window or window < 1:
        raise ValueError('window must be a positive integer')
    if not (0 < delta < 1 and 0 < eta < 1):
        raise ValueError('delta and eta must be strictly between zero and one')
    window = int(window)
    if binom.sf(window - 1, window, 1 - delta) > eta:
        return None
    lo, hi = 1, window
    while lo < hi:
        mid = (lo + hi) // 2
        if binom.sf(mid - 1, window, 1 - delta) <= eta:
            hi = mid
        else:
            lo = mid + 1
    return lo


def margin_path(demand, forecast, horizon, window, delta, eta=.05, start=0):
    """Return issue-time PAC margins for ticks start..len(demand)-1.

    At tick t the W scores are max(y[j]-forecast[j-horizon], 0),
    j=t-W+1..t. All zeros remain in the window. Only mature truth is used.
    A full preceding calibration prefix is required; no rank clipping or warmup
    fallback silently replaces the specified PAC statistic.
    """
    y, q = np.asarray(demand, dtype=float), np.asarray(forecast, dtype=float)
    if y.ndim != 1 or q.shape != y.shape or not len(y):
        raise ValueError('demand and forecast must be equal-length nonempty vectors')
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(q)) or np.any(y < 0) or np.any(q < 0):
        raise ValueError('demand and forecast must be finite and nonnegative')
    if isinstance(horizon, bool) or int(horizon) != horizon or horizon < 1:
        raise ValueError('horizon must be a positive integer')
    rank = pac_rank(window, delta, eta)
    if rank is None:
        raise ValueError('No finite PAC rank; enlarge the window before replay')
    horizon, window = int(horizon), int(window)
    if isinstance(start, bool) or int(start) != start or not window + horizon - 1 <= start < len(y):
        raise ValueError('start must leave W mature preceding scores and an evaluation tick')
    start = int(start)
    scores = np.maximum(y[horizon:] - q[:-horizon], 0.)
    ordered = sorted(scores[start - horizon - window + 1:start - horizon + 1])
    result = np.empty(len(y) - start)
    # ponytail: sorted-list insertion is O(W); avoids O(T*W) sliding-window copies.
    for t in range(start, len(y)):
        result[t - start] = ordered[rank - 1]
        if t + 1 < len(y):
            old = scores[t - horizon - window + 1]
            ordered.pop(bisect_left(ordered, old))
            insort(ordered, scores[t - horizon + 1])
    return result


def margin_grid(demand, forecast, horizon, windows, deltas, eta=.05, start=0):
    """PAC margins keyed by (W, delta), sharing one rolling window per W.

    None/saturated ranks are omitted, matching the official admissible-window
    rule. Every returned path uses the same mature-score convention as
    margin_path; omitted candidates must be counted explicitly by the caller.
    """
    y, q = np.asarray(demand, dtype=float), np.asarray(forecast, dtype=float)
    if y.ndim != 1 or q.shape != y.shape or not len(y):
        raise ValueError('demand and forecast must be equal-length nonempty vectors')
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(q)) or np.any(y < 0) or np.any(q < 0):
        raise ValueError('demand and forecast must be finite and nonnegative')
    if isinstance(horizon, bool) or int(horizon) != horizon or horizon < 1:
        raise ValueError('horizon must be a positive integer')
    if isinstance(start, bool) or int(start) != start or not 0 <= start < len(y):
        raise ValueError('start must be an evaluation tick')
    if not 0 < eta < 1:
        raise ValueError('eta must be strictly between zero and one')
    horizon, start = int(horizon), int(start)
    scores = np.maximum(y[horizon:] - q[:-horizon], 0.)
    result = {}
    deltas = tuple(dict.fromkeys(deltas))
    for window in dict.fromkeys(windows):
        candidates = []
        for delta in deltas:
            rank = pac_rank(window, delta, eta)
            if rank is not None and rank <= window - 1:
                candidates.append((delta, rank))
        if not candidates:
            continue
        window = int(window)
        if start < window + horizon - 1:
            raise ValueError('start must leave W mature preceding scores')
        paths = _margin_columns(scores, horizon, window,
                                np.asarray([rank for _, rank in candidates], dtype=np.int64), start)
        for row, (delta, _) in enumerate(candidates):
            result[window, delta] = paths[row]
    return result
