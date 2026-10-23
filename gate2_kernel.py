"""Compiled replay of the existing CDT/BACC-lite and Actuator mechanisms.

Records columns: demand, loss, ready, paid, requested, parameter_used.
Only the caller decides the evaluation protocol; this module runs no experiment.
"""

import math
from pathlib import Path
import sys

import numpy as np

_dependencies = Path(__file__).resolve().parent / 'work/pydeps'
if _dependencies.exists():
    sys.path.insert(0, str(_dependencies))
from numba import njit


@njit(cache=True)
def _replay(y, q, capacity, h_actual, h_assumed, maximum, rate, stable_steps,
            kind, epsilon, parameter, cal_scores, targets, record):
    length = len(y)
    records = np.empty((length if record else 0, 6), dtype=np.float64)
    pending = np.zeros(h_actual + 1, dtype=np.int64)
    pending_up = down_next = 0
    recommendations = np.zeros(stable_steps, dtype=np.int64)
    replicas = 1
    violations = paid_cost = ready_cost = actions = 0
    value = minimum_value = 0.

    # BACC uses bounded rings and a sorted array, preserving all 720 scores.
    score_count = min(len(cal_scores), 720)
    scores = np.empty(720, dtype=np.float64)
    for j in range(score_count):
        scores[j] = min(20., max(.05, cal_scores[len(cal_scores) - score_count + j]))
    ordered = np.empty(720, dtype=np.float64)
    ordered[:score_count] = np.sort(scores[:score_count])
    score_cursor = 0
    recent = np.zeros(1440, dtype=np.int64)
    recent_sum = 0
    upper_due = np.empty(h_assumed, dtype=np.float64)
    alpha_due = np.empty(h_assumed, dtype=np.float64)
    pace = epsilon
    misses = alpha_units = 0.
    units_by_miss = np.zeros(2, dtype=np.int64)
    bad_by_miss = np.zeros(2, dtype=np.int64)

    for t in range(length):
        slot = t % len(pending)
        replicas += pending[slot] + down_next
        pending_up -= pending[slot]
        pending[slot] = 0
        down_next = 0
        ready = replicas
        paid = ready + pending_up
        loss = int(y[t] > capacity * ready)
        violations += loss

        used = 0.
        if kind == 0:
            target = max(1, min(maximum, math.ceil(max(0., q[t]) / capacity - value)))
            used = value
        elif kind == 1:
            target = math.ceil(y[t] / (capacity * parameter))
            used = parameter
        elif kind == 2:
            recent_slot = t % 1440
            recent_sum += loss - recent[recent_slot]
            recent[recent_slot] = loss
            if t >= h_assumed:
                new_score = min(20., max(.05, (y[t] + 1) / (q[t - h_assumed] + 1)))
                if score_count == 720:
                    old = scores[score_cursor]
                    pos = np.searchsorted(ordered[:score_count], old)
                    for j in range(pos, score_count - 1):
                        ordered[j] = ordered[j + 1]
                    score_count -= 1
                    scores[score_cursor] = new_score
                    score_cursor = (score_cursor + 1) % 720
                else:
                    scores[score_count] = new_score
                pos = np.searchsorted(ordered[:score_count], new_score)
                for j in range(score_count, pos, -1):
                    ordered[j] = ordered[j - 1]
                ordered[pos] = new_score
                score_count += 1
                old_slot = t % h_assumed
                missed = int(y[t] > upper_due[old_slot])
                units_by_miss[missed] += 1
                bad_by_miss[missed] += loss
                misses += missed
                alpha_units += alpha_due[old_slot]

            pacing_error = max(recent_sum / min(t + 1, 1440), violations / (t + 1)) - epsilon
            gain = parameter * (1. if pacing_error > 0 else .2)
            pace = min(.9, max(.2 * epsilon, pace - gain * pacing_error))
            alpha = pace
            remaining = length - t - 1
            if remaining:
                reserve = loss * min(max(0, h_assumed - 1), remaining)
                affordable = min(1., max(0., (math.floor(epsilon * length) - violations - reserve) / remaining))
                p0 = (bad_by_miss[0] + .5) / (units_by_miss[0] + 1)
                p1 = (bad_by_miss[1] + .5) / (units_by_miss[1] + 1)
                if affordable < p0:
                    max_miss = 0.
                elif affordable < epsilon and p1 > p0:
                    max_miss = min(1., max(0., (affordable - p0) / (p1 - p0)))
                else:
                    max_miss = 1.
                conversion = min(10., max(.1, (misses + .5) / (alpha_units + .5)))
                alpha = max(.1 * epsilon, min(alpha, max_miss / conversion))
            rank = min(score_count, math.ceil((score_count + 1) * (1 - alpha)))
            factor = ordered[rank - 1] if score_count else 1.
            upper = max(0., (q[t] + 1) * factor - 1)
            upper_due[t % h_assumed] = upper
            alpha_due[t % h_assumed] = alpha
            target = math.ceil(upper / capacity)
            used = alpha
        else:
            target = targets[t]

        wanted = math.ceil(min(maximum, max(1., target)))
        recommendations[t % stable_steps] = wanted
        projected = ready + pending_up
        if wanted > projected:
            delta = min(rate, wanted - projected)
            pending[(t + h_actual) % len(pending)] += delta
            pending_up += delta
            actions += 1
        elif pending_up == 0:
            stabilized = 1
            for j in range(min(t + 1, stable_steps)):
                stabilized = max(stabilized, recommendations[j])
            delta = max(-rate, min(0, stabilized - ready))
            if delta:
                down_next = delta
                actions += 1
        if kind == 0:
            value += parameter * (epsilon - loss)
            minimum_value = min(minimum_value, value)
        paid_cost += paid
        ready_cost += ready
        if record:
            records[t, 0] = y[t]
            records[t, 1] = loss
            records[t, 2] = ready
            records[t, 3] = paid
            records[t, 4] = target
            records[t, 5] = used
    return violations, paid_cost, ready_cost, actions, value, minimum_value, records


def replay(y, q, capacity, h_actual, h_assumed, maximum, rate, stable_seconds,
           kind, epsilon, parameter, cal_scores=(), targets=(), record=False):
    """Replay kind 0=CDT, 1=HPA, 2=BACC-lite, 3=precomputed requests.

    h_actual/h_assumed are service-index lags, already including the boundary.
    Return (violations, paid_cost, ready_cost, actions, final_lambda, min_lambda,
    records). BACC's calibration window is 720 and recent-loss window is 1440.
    """
    y, q = np.asarray(y, dtype=np.float64), np.asarray(q, dtype=np.float64)
    cal_scores, targets = np.asarray(cal_scores, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    if y.ndim != 1 or q.shape != y.shape or len(y) == 0:
        raise ValueError('y and q must be equal-length nonempty vectors')
    if np.any(~np.isfinite(y)) or np.any(~np.isfinite(q)) or np.any(y < 0) or np.any(q < 0):
        raise ValueError('Demand and predictions must be finite and nonnegative')
    if any(int(v) != v or v < 1 for v in (h_actual, h_assumed, maximum, rate)):
        raise ValueError('Lags, maximum, and rate must be positive integers')
    if not (math.isfinite(capacity) and capacity > 0 and 0 < epsilon < 1 and
            math.isfinite(stable_seconds) and stable_seconds >= 0 and kind in (0, 1, 2, 3)):
        raise ValueError('Invalid replay settings')
    if not math.isfinite(parameter) or (kind != 3 and parameter <= 0):
        raise ValueError('Invalid controller parameter')
    if cal_scores.ndim != 1 or np.any(~np.isfinite(cal_scores)):
        raise ValueError('Calibration scores must be a finite vector')
    if kind == 3 and (targets.shape != y.shape or np.any(~np.isfinite(targets))):
        raise ValueError('Precomputed targets must match demand and be finite')
    return _replay(y, q, float(capacity), int(h_actual), int(h_assumed), int(maximum), int(rate),
                   max(1, math.ceil(stable_seconds / 60)), int(kind), float(epsilon),
                   float(parameter), cal_scores, targets, bool(record))
