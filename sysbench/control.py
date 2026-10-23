"""Pure v3 tick control and passive monitoring; no container or trace IO.

Counts use a five-second interval: capacity_count = 5 * calibrated capacity_rps.
The revised contract is R=8, a=2, dmax=8 (35 seconds), epsilon=.05 and eta=.5.
Its raw bound at T=240 is 1.0333, not an exact 5% service-risk guarantee.
"""

from collections import deque
import math
import time


def count(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(name + " must be a nonnegative integer")
    return value


def positive(value, name):
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be finite and positive")
    return value


def hpa_request(y_count, capacity_rps, utilization):
    """Unclipped integer HPA proposal, in replicas, from a five-second count."""
    return math.ceil(count(y_count, "arrival count") /
                     (5 * positive(capacity_rps, "capacity_rps") * positive(utilization, "utilization")))


def certificate(ticks, loss_total, *, lambda_final=None, lambda_min=None,
                epsilon=.05, eta=.5, maximum=8, minimum=1, rate=2,
                tick_seconds=5, startup_bound_seconds=35):
    """Arithmetic only. Applicability additionally requires the execution contract.

    Baselines pass no lambda and retain the same display-bound fields with an
    unavailable update identity. The original bound is never replaced by its
    [0,1] display clamp. Both include the initial lambda_0=0 convention.
    """
    count(ticks, "ticks")
    count(loss_total, "loss_total")
    if loss_total > ticks or not 0 < epsilon < 1:
        raise ValueError("Invalid cumulative loss or target")
    positive(eta, "eta")
    if not 1 <= count(minimum, "minimum") <= count(maximum, "maximum"):
        raise ValueError("Invalid replica bounds")
    if not count(rate, "rate"):
        raise ValueError("Scale-up rate must be positive")
    dmax = 1 + math.ceil(positive(startup_bound_seconds, "startup bound") /
                         positive(tick_seconds, "tick_seconds"))
    m = math.ceil((maximum - minimum) / rate)
    settling = dmax + m - 1 if m else 0
    window = max(1, math.ceil(settling / epsilon))
    raw = epsilon + (maximum / eta + window) / ticks if ticks else None
    if (lambda_final is None) != (lambda_min is None):
        raise ValueError("Both lambda summaries are required together")
    if lambda_final is not None and not (math.isfinite(lambda_final) and math.isfinite(lambda_min)
                                         and lambda_min <= min(0., lambda_final)):
        raise ValueError("Invalid lambda summaries")
    residual = (loss_total - (ticks * epsilon - lambda_final / eta)
                if lambda_final is not None else None)
    return dict(ticks=ticks, loss_total=loss_total, loss_rate=loss_total / ticks if ticks else None,
                lambda_final=lambda_final, lambda_min=lambda_min, lambda_safe=-maximum,
                dmax=dmax, m=m, L=settling, K=window, bound_raw=raw,
                bound_clamped=min(1., max(0., raw)) if ticks else None,
                identity_residual=residual,
                arithmetic_ok=abs(residual) <= 1e-9 if residual is not None else None)


class ContractMonitor:
    """Passive observer shared by all arms; flags latch their first observation.

    pod_requested is called when the complete create request is first initiated,
    before invoking the runtime API. pod_ready uses the first observed Ready time;
    injected sleep alone is never a startup measurement. Requests here are actual
    emitted commands or observed external mutations, not an unclipped proposal.
    """

    def __init__(self, capacity_rps, *, clock=time.monotonic, maximum=8,
                 startup_bound_seconds=35):
        self.capacity_count = 5 * positive(capacity_rps, "capacity_rps")
        if not count(maximum, "maximum"):
            raise ValueError("Maximum must be positive")
        self.maximum = maximum
        self.startup_bound_seconds = positive(startup_bound_seconds, "startup bound")
        self.clock, self.last_now = clock, -math.inf
        self.pending, self.startup_seconds, self.flags = {}, {}, {}
        self.ticks = self.loss_total = 0

    def now(self, when=None):
        value = self.clock() if when is None else when
        if not math.isfinite(value) or value < self.last_now:
            raise ValueError("A finite nondecreasing monotonic clock is required")
        self.last_now = value
        return value

    def invalidate(self, name, *, when=None):
        observed = self.now(when)
        self.flags.setdefault(name, dict(observed_at=observed, tick=self.ticks))

    def pod_requested(self, pod_id, *, when=None):
        observed = self.now(when)
        if not pod_id or pod_id in self.pending or pod_id in self.startup_seconds:
            raise ValueError("A new immutable pod identity is required")
        self.pending[pod_id] = observed

    def pod_ready(self, pod_id, *, when=None):
        observed = self.now(when)
        if pod_id not in self.pending:
            raise ValueError("Ready must refer to a recorded complete create request")
        duration = observed - self.pending.pop(pod_id)
        self.startup_seconds[pod_id] = duration
        if duration > self.startup_bound_seconds:
            self.invalidate("startup_bound", when=observed)
        return duration

    def observe_state(self, *, runtime_cap, external_request=None, when=None):
        observed = self.now(when)
        count(runtime_cap, "runtime_cap")
        if runtime_cap < self.maximum:
            self.invalidate("runtime_cap", when=observed)
        if external_request is not None:
            count(external_request, "external_request")
            if external_request > self.maximum:
                self.invalidate("request_above_maximum", when=observed)
            elif external_request < 1:
                self.invalidate("request_below_minimum", when=observed)
        if any(observed - started > self.startup_bound_seconds for started in self.pending.values()):
            self.invalidate("startup_bound", when=observed)

    def observe_tick(self, index, y_count, n_start, *, runtime_cap=8, external_request=None, when=None):
        if count(index, "tick index") != self.ticks:
            raise ValueError("Tick indices must be consecutive from zero")
        count(y_count, "arrival count")
        count(n_start, "tick-start ready count")
        observed = self.now(when)
        self.observe_state(runtime_cap=runtime_cap, external_request=external_request, when=observed)
        if y_count > self.capacity_count * self.maximum:
            self.invalidate("demand_envelope", when=observed)
        loss = int(y_count > self.capacity_count * n_start)
        self.ticks += 1
        self.loss_total += loss
        return loss

    def snapshot(self, *, lambda_final=None, lambda_min=None):
        result = certificate(self.ticks, self.loss_total, lambda_final=lambda_final,
                             lambda_min=lambda_min, maximum=self.maximum,
                             startup_bound_seconds=self.startup_bound_seconds)
        result.update(contract_valid=not self.flags,
                      certificate_status="NOT_APPLICABLE" if lambda_final is None else
                      "OUTSIDE_OBSERVED_CONTRACT" if self.flags else "CONDITIONAL_ON_EXECUTION_CONTRACT",
                      first_flags={key: dict(value) for key, value in self.flags.items()})
        return result


class TickController:
    """Finish a tick with old-lambda emission, then unprojected actual-loss feedback.

    ready_now + inflight is the runtime's projected count at decision time.
    Scale-up adds at most two; any inflight pod forbids scale-down/cancellation.
    When none is inflight, scale-down needs five recommendations and uses their
    maximum, without adding an unrequested scale-down rate limit.
    Actual container mutations belong to the injected emit(target) callback.
    """

    def __init__(self, monitor, *, wrapped=True):
        self.monitor, self.wrapped = monitor, wrapped
        self.value = self.minimum_value = 0.
        self.recommendations = deque(maxlen=5)

    def finish_tick(self, index, y_count, n_start, base_request, *, ready_now, inflight,
                    emit, runtime_cap=8, external_request=None, when=None):
        count(ready_now, "decision-time ready count")
        count(inflight, "inflight count")
        projected = ready_now + inflight
        if not 1 <= projected <= self.monitor.maximum:
            raise ValueError("Cannot safely mutate an out-of-range projected runtime count")
        if not math.isfinite(base_request) or base_request < 0:
            raise ValueError("Base proposal must be finite and nonnegative")
        used = self.value
        raw = base_request - used if self.wrapped else base_request
        # Integer R makes ceil(clip(raw)) equivalent to clip(ceil(raw)).
        requested = math.ceil(min(self.monitor.maximum, max(1., raw)))
        recommendations = (*self.recommendations, requested)[-5:]
        command = projected
        if requested > projected:
            command = min(requested, projected + 2)
        elif not inflight and len(recommendations) == 5:
            command = min(projected, max(recommendations))
        loss = self.monitor.observe_tick(index, y_count, n_start, runtime_cap=runtime_cap,
                                         external_request=external_request, when=when)
        self.recommendations.append(requested)
        error = None
        try:
            emit(command)  # Runtime dispatch occurs while self.value is still lambda_t.
        except Exception as failure:
            error = type(failure).__name__ + ": " + str(failure)
            self.monitor.invalidate("command_error", when=self.monitor.last_now)
        if self.wrapped:
            self.value = used + .5 * (.05 - loss)
            self.minimum_value = min(self.minimum_value, self.value)
        result = dict(index=index, y_count=y_count, n_start=n_start, ready_now=ready_now,
                      inflight=inflight, base_request=base_request,
                      lambda_t=used if self.wrapped else None,
                      unclipped_request=raw, saturated=raw < 1 or raw > self.monitor.maximum,
                      request=requested, command=command, issued=error is None, error=error, loss=loss)
        result.update(self.monitor.snapshot(lambda_final=self.value if self.wrapped else None,
                                           lambda_min=self.minimum_value if self.wrapped else None))
        return result
