"""Causal calibration and actuation primitives; not a full benchmark implementation."""

from collections import deque
from dataclasses import dataclass
import argparse
import math
import tomllib


def conformal_quantile(scores, alpha):
    """One-sided split-conformal rank, with explicit extended-real endpoints."""
    if alpha <= 0:
        return math.inf
    if alpha >= 1:
        return -math.inf
    ordered = sorted(scores)
    rank = math.ceil((len(ordered) + 1) * (1 - alpha))
    return ordered[rank - 1] if 0 < rank <= len(ordered) else math.inf


@dataclass(frozen=True)
class Forecast:
    issued: int
    due: int
    base: float
    upper: float
    alpha: float


class DelayedACI:
    """Issue before reveal at each tick; e_i is first available at i+h.

    A reveal at t updates the state used at t+1. Scores and issued bounds are
    saved separately, so a future truth cannot change an already issued bound.
    Clipping is practical; its projection remainder is part of the certificate.
    """

    def __init__(self, alpha=.05, gamma=.01, horizon=2, window=1000,
                 clip=(.001, .999), scores=()):
        if not (0 < alpha < 1 and math.isfinite(gamma) and gamma > 0 and
                isinstance(horizon, int) and horizon >= 1 and isinstance(window, int) and window >= 1):
            raise ValueError("Invalid ACI configuration")
        if clip is not None and not (0 < clip[0] <= alpha <= clip[1] < 1):
            raise ValueError("Clipping bounds must contain the target")
        scores = tuple(scores)
        if any(not math.isfinite(x) for x in scores):
            raise ValueError("Calibration scores must be finite")
        self.target = self.initial = self.alpha = alpha
        self.gamma, self.horizon, self.clip = gamma, horizon, clip
        self.scores = deque(scores, maxlen=window)
        self.pending, self.errors = {}, []
        self.projection_sum = 0.0
        self.last_issue = self.last_reveal = -1

    def issue(self, t, base):
        if t <= self.last_issue or t <= self.last_reveal or not math.isfinite(base):
            raise ValueError("Issue times must increase and predictions must be finite")
        prediction = Forecast(t, t + self.horizon, base,
                              base + conformal_quantile(self.scores, self.alpha), self.alpha)
        self.pending[prediction.due] = prediction
        self.last_issue = t
        return prediction

    def reveal(self, t, truth):
        if t <= self.last_reveal or not math.isfinite(truth):
            raise ValueError("Reveal times must increase and truths must be finite")
        if any(due < t for due in self.pending):
            raise ValueError("A matured forecast was skipped")
        self.last_reveal = t
        prediction = self.pending.pop(t, None)
        if prediction is None:
            return None
        error = int(truth > prediction.upper)
        raw = self.alpha + self.gamma * (self.target - error)
        self.alpha = raw if self.clip is None else min(self.clip[1], max(self.clip[0], raw))
        self.projection_sum += self.alpha - raw
        self.errors.append(error)
        self.scores.append(truth - prediction.base)
        return error

    def certificate(self):
        """Exact identity over revealed forecasts, not an unconditional SLO bound."""
        n = len(self.errors)
        if not n:
            raise ValueError("No matured forecasts")
        gap = sum(self.errors) / n - self.target
        accounted = (self.initial - self.alpha + self.projection_sum) / (self.gamma * n)
        return n, gap, accounted


class HysteresisGate:
    def __init__(self, hold=5, ok=3, max_error=.1, max_margin=2.):
        if hold < 1 or ok < 1 or not 0 <= max_error <= 1 or max_margin <= 0:
            raise ValueError("Invalid gate configuration")
        self.hold, self.ok = hold, ok
        self.max_error, self.max_margin = max_error, max_margin
        self.fallback, self.until, self.good, self.last = False, -1, 0, -1

    def step(self, t, recent_error, relative_margin, drift):
        if t <= self.last or not 0 <= recent_error <= 1 or math.isnan(relative_margin):
            raise ValueError("Invalid gate observation")
        self.last = t
        bad = recent_error > self.max_error or relative_margin > self.max_margin or drift
        self.good = 0 if bad else self.good + 1
        if bad:
            self.fallback, self.until = True, t + self.hold
        elif self.fallback and t >= self.until and self.good >= self.ok:
            self.fallback = False
        return self.fallback


class Actuator:
    """Delayed scale-up, stabilized scale-down, and bounded per-tick commands.

    Call begin(t) before serving tick t, then command(t, target) after observing
    it. Pending scale-ups reserve replicas. Scale-down waits until they finish.
    """

    def __init__(self, step_seconds=60, delay_seconds=90, capacity=10.,
                 minimum=1, maximum=32, max_change=4, stable_seconds=300):
        if not (step_seconds > 0 and delay_seconds >= 0 and capacity > 0 and
                1 <= minimum <= maximum and max_change >= 1 and stable_seconds >= 0):
            raise ValueError("Invalid actuator configuration")
        self.horizon = max(1, math.ceil(delay_seconds / step_seconds))
        self.c, self.minimum, self.maximum, self.limit = capacity, minimum, maximum, max_change
        self.replicas, self.pending = minimum, []
        self.recommendations = deque(maxlen=max(1, math.ceil(stable_seconds / step_seconds)))
        self.tick = -1
        self.commanded = False

    def begin(self, t):
        if t != self.tick + 1:
            raise ValueError("Actuation ticks must be consecutive")
        self.tick, self.commanded = t, False
        self.replicas += sum(delta for due, delta in self.pending if due == t)
        self.pending = [(due, delta) for due, delta in self.pending if due > t]
        assert self.minimum <= self.replicas <= self.maximum
        return self.replicas * self.c

    def command(self, t, target):
        if t != self.tick or self.commanded or math.isnan(target):
            raise ValueError("Exactly one command is allowed after begin(t)")
        self.commanded = True
        wanted = min(self.maximum, max(self.minimum, target))
        wanted = math.ceil(wanted)
        self.recommendations.append(wanted)
        projected = self.replicas + sum(delta for _, delta in self.pending)
        if wanted > projected:
            self.pending.append((t + self.horizon, min(self.limit, wanted - projected)))
        elif not self.pending:
            stabilized = max(self.recommendations)
            delta = max(-self.limit, min(0, stabilized - self.replicas))
            if delta:
                self.pending.append((t + 1, delta))


def diagnostics(config):
    settings = config['calibration']
    horizon = Actuator(**config['actuator']).horizon
    rows = []
    for name, clipping in [('projected', tuple(settings['clip'])), ('unprojected reference', None)]:
        model = DelayedACI(settings['alpha'], settings['gamma'], horizon,
                           settings['window'], clipping, [0.] * settings['window'])
        n = config['diagnostic']['forecasts']
        for t in range(n + model.horizon):
            if t < n:
                model.issue(t, 0.)
            model.reveal(t, float(t + 1))
        count, gap, accounted = model.certificate()
        assert count == n and math.isclose(gap, accounted, abs_tol=1e-10)
        rows.append((name, count, sum(model.errors) / count, gap, model.projection_sum))
    print('# Core diagnostics\n\nDeterministic increasing labels; engineering checks, not trace benchmarks.\n')
    print('| Branch | Forecasts | Miscoverage | Gap from target | Projection sum |')
    print('|---|---:|---:|---:|---:|')
    for name, count, error, gap, projection in rows:
        print(f'| {name} | {count} | {error:.6f} | {gap:.6f} | {projection:.6f} |')
    print('\nBoth exact telescoping identities passed. The clipped branch can keep missing indefinitely.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/core.toml')
    args = parser.parse_args()
    with open(args.config, 'rb') as stream:
        diagnostics(tomllib.load(stream))
