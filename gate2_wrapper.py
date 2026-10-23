"""P protocol candidate: a loss-feedback wrapper around fixed base suggestions.

No trace runner, training, parameter selection, or full-BACC modification lives
here. Bounds are conditional on dynamically reachable uncancelled Actuator
queues, y <= capacity*maximum, and actual service lag <= the declared bound.
"""

from collections import deque
import math

from cdt_pilot import safety_constants
from paper4 import Actuator


ETA_CANDIDATES = (.02, .05, .1, .2, .5)  # User candidates; no value is selected here.


def nonnegative(value, name):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def positive(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def wrapped_request(base_request, value, minimum, maximum):
    """clip(ceil(base_request-value), minimum, maximum), with integer bounds.

    base_request is the base's unbounded desired replica proposal, possibly
    already ceiled by that base, but never pre-clipped to the actuator's cap.
    """
    nonnegative(base_request, "base_request")
    if not math.isfinite(value):
        raise ValueError("lambda must be finite")
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (minimum, maximum)) or not 1 <= minimum <= maximum:
        raise ValueError("Replica bounds must be positive integers")
    # Clipping before ceil is identical for integer bounds and avoids ceil(inf).
    return math.ceil(min(maximum, max(minimum, base_request - value)))


class FeedbackWrapper:
    """Only suggestions enter the wrapper; bases receive no actuator mutation API.

    A physical startup D uses Actuator(delay_seconds=D+step_seconds), preserving
    minute-end issue semantics. External mutation of its pending queue is outside
    the contract. Cancel-capable or alternative actuators require a new proof.
    This scalar interface accepts y/q-history proposals supplied before begin;
    policies requiring the post-begin Ready state are not supported here.
    """

    def __init__(self, plant, epsilon, eta, *, cancel_pending_up=False):
        if type(plant) is not Actuator or cancel_pending_up or getattr(plant, "cancel_pending_up", False):
            raise ValueError("Only the existing non-cancelling Actuator is supported")
        wrapped_request(0., 0., plant.minimum, plant.maximum)
        if (isinstance(plant.limit, bool) or not isinstance(plant.limit, int)
                or not isinstance(plant.horizon, int)):
            raise ValueError("Rate and service lag must be integers")
        if not math.isfinite(epsilon) or not 0 < epsilon < 1:
            raise ValueError("epsilon must be strictly between zero and one")
        self.plant, self.epsilon, self.eta = plant, epsilon, positive(eta, "eta")
        self.value = 0.

    def constants(self, delay_bound):
        """Return (lambda_safe, L, K); demand-envelope checking remains required."""
        if (isinstance(delay_bound, bool) or not isinstance(delay_bound, int)
                or delay_bound < self.plant.horizon):
            raise ValueError("Actual service lag is outside the stated delay contract")
        return safety_constants(self.plant.maximum, self.plant.minimum,
                                self.plant.limit, delay_bound, self.epsilon)

    def step(self, t, demand, base_request):
        if isinstance(t, bool) or not isinstance(t, int):
            raise ValueError("Service ticks must be integers")
        nonnegative(demand, "demand")
        used = self.value
        request = wrapped_request(base_request, used, self.plant.minimum, self.plant.maximum)
        effective = self.plant.begin(t)
        ready = self.plant.replicas
        paid = ready + sum(max(0, delta) for _, delta in self.plant.pending)
        loss = int(demand > effective)
        next_value = used + self.eta * (self.epsilon - loss)
        if not math.isfinite(next_value):
            raise OverflowError("Non-projected lambda exceeded floating-point range")
        self.plant.command(t, request)  # Issue using lambda_t before actual-loss feedback.
        self.value = next_value
        return dict(ready=ready, paid=paid, loss=loss, request=request,
                    used_lambda=used, value=self.value,
                    in_envelope=demand <= self.plant.c * self.plant.maximum)


def hpa_request(demand, capacity, utilization):
    """The v1 HPA suggestion; frozen utilization is supplied by the caller."""
    return math.ceil(nonnegative(demand, "demand") /
                     (positive(capacity, "capacity") * positive(utilization, "utilization")))


def point_margin_request(prediction, capacity, margin):
    """The v1 additive replica-margin suggestion, without retuning its margin."""
    return math.ceil(nonnegative(prediction, "prediction") / positive(capacity, "capacity")
                     + nonnegative(margin, "margin"))


def pac_requests(demand, prediction, capacity, horizon, window, delta,
                 confidence_failure=.05, start=0):
    """Reuse v1 mature-score PAC-h and its formal-grid availability guard."""
    import numpy as np
    from gate2_pac import margin_path, pac_rank

    positive(capacity, "capacity")
    rank = pac_rank(window, delta, confidence_failure)
    if rank is None:
        raise ValueError("No finite PAC rank; candidate is unavailable")
    if rank > window - 1:
        raise ValueError("Saturated PAC rank; v1 requires rank <= window-1")
    margin = margin_path(demand, prediction, horizon, window, delta,
                         eta=confidence_failure, start=start)
    return np.ceil((np.asarray(prediction, dtype=float)[start:] + margin) / capacity)


class BACCCal:
    """BACC-cal: fixed-alpha ratio calibration only; not full BACC.

    Preserve v1 ratio clipping [.05,20] and finite rank cap. Mature calibration
    residuals change the score window; actual capacity loss has no input path.
    There is no pacing, remaining budget, miss/loss conversion, or second loop.
    """

    def __init__(self, capacity, alpha, horizon, window, scores):
        self.capacity = positive(capacity, "capacity")
        if not math.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("Fixed alpha must be strictly between zero and one")
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in (horizon, window)):
            raise ValueError("Horizon and calibration window must be positive integers")
        scores = tuple(scores)
        if not scores or any(not math.isfinite(s) for s in scores):
            raise ValueError("Initial mature finite calibration scores are required")
        self.alpha, self.horizon = alpha, horizon
        self.scores = deque((min(20., max(.05, float(s))) for s in scores), maxlen=window)
        self.pending, self.last = {}, -1

    def suggest(self, t, demand, prediction):
        nonnegative(demand, "demand")
        nonnegative(prediction, "prediction")
        if isinstance(t, bool) or not isinstance(t, int) or t != self.last + 1:
            raise ValueError("BACC-cal issue times must be consecutive from zero")
        old = self.pending.get(t)
        candidate_scores = list(self.scores)
        if old is not None:
            new_score = min(20., max(.05, (demand + 1) / (old + 1)))
            candidate_scores = (candidate_scores + [new_score])[-self.scores.maxlen:]
        ordered = sorted(candidate_scores)
        rank = min(len(ordered), math.ceil((len(ordered) + 1) * (1 - self.alpha)))
        upper = max(0., (prediction + 1) * ordered[rank - 1] - 1)
        nonnegative(upper, "calibrated upper")
        if old is not None:
            self.pending.pop(t)
            self.scores.append(new_score)
        self.pending[t + self.horizon] = prediction
        self.last = t
        return math.ceil(upper / self.capacity)
