"""Seeded Poisson schedules and injected open-loop replay; no HTTP client or IO."""

import asyncio
import inspect
import math
import random


def poisson_timestamps(rates_rps, *, seed, tick_seconds=5.):
    """Immutable relative seconds for a piecewise-constant Poisson arrival rate.

    Each input is requests/second during one tick, not an arrival count. Independent
    exponential increments are generated inside each half-open tick interval.
    The local RNG never reads or changes Python's global random state.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("A fixed integer seed is required")
    if not math.isfinite(tick_seconds) or tick_seconds <= 0:
        raise ValueError("tick_seconds must be finite and positive")
    rng, times = random.Random(seed), []
    for tick, rate in enumerate(rates_rps):
        if isinstance(rate, bool) or not math.isfinite(rate) or rate < 0:
            raise ValueError("Rates must be finite nonnegative requests/second")
        if not rate:
            continue
        current, end = tick * tick_seconds, (tick + 1) * tick_seconds
        while True:
            following = current + rng.expovariate(rate)
            if following <= current:
                raise ValueError("Arrival rate exceeds timestamp precision")
            if following >= end:
                break
            times.append(following)
            current = following
    return tuple(times)


async def dispatch(schedule, send, on_result, *, timeout, start_at=None,
                   clock=None, sleep=asyncio.sleep):
    """Replay fixed offsets against absolute monotonic deadlines.

    send(index) is asynchronous; on_result(record) may be sync or async. The
    recorded dispatch time is entry into the provided send operation, not a
    claimed network-wire/server timestamp. Responses never pace future starts.
    Only outstanding tasks are retained; callbacks must perform nonblocking
    aggregation/enqueueing, with disk IO outside the event loop. Their exceptions
    are counted and summarized without stopping
    later arrivals. No semaphore, rate adaptation or response-result list is used.
    """
    schedule = tuple(schedule)  # Reuses a tuple unchanged across controller arms.
    if any(not math.isfinite(t) or t < 0 for t in schedule) or any(
            a > b for a, b in zip(schedule, schedule[1:])):
        raise ValueError("Schedule offsets must be finite, nonnegative and ordered")
    if not callable(send) or not callable(on_result):
        raise ValueError("send and on_result callbacks are required")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("An explicit positive request timeout is required")
    clock = clock or asyncio.get_running_loop().time
    last = -math.inf

    def observed_time():
        nonlocal last
        now = clock()
        if not math.isfinite(now) or now < last:
            raise ValueError("A finite nondecreasing monotonic clock is required")
        last = now
        return now

    origin = observed_time() if start_at is None else start_at
    if not math.isfinite(origin):
        raise ValueError("start_at must use the finite monotonic clock domain")
    stats = dict(scheduled=len(schedule), dispatched=0, ok=0, timeout=0,
                 error=0, cancelled=0, callback_errors=0, first_callback_error=None)
    pending = set()

    async def complete(index, deadline):
        record = dict(index=index, scheduled_offset=schedule[index], planned_at=deadline,
                      actual_dispatch_at=None, completed_at=None, status="OK", error=None, result=None)
        cancelled = False
        async def attempt():
            # Timestamp the coroutine's real entry, not merely create_task time.
            record["actual_dispatch_at"] = observed_time()
            stats["dispatched"] += 1
            return await send(index)
        try:
            record["result"] = await asyncio.wait_for(attempt(), timeout)
        except asyncio.TimeoutError:
            record.update(status="TIMEOUT", error="TimeoutError")
        except asyncio.CancelledError:
            cancelled = True
            record.update(status="CANCELLED", error="CancelledError")
        except Exception as failure:
            record.update(status="ERROR", error=type(failure).__name__ + ": " + str(failure))
        try:
            record["completed_at"] = observed_time()
        except Exception as failure:
            record.update(status="ERROR", error=type(failure).__name__ + ": " + str(failure))
        stats[record["status"].lower()] += 1
        try:
            returned = on_result(record)
            if inspect.isawaitable(returned):
                await returned
        except (Exception, asyncio.CancelledError) as failure:
            stats["callback_errors"] += 1
            if stats["first_callback_error"] is None:
                stats["first_callback_error"] = type(failure).__name__ + ": " + str(failure)
        if cancelled:
            raise asyncio.CancelledError

    try:
        for index, offset in enumerate(schedule):
            deadline = origin + offset
            if not math.isfinite(deadline):
                raise ValueError("Absolute arrival deadline overflow")
            while True:
                remaining = deadline - observed_time()
                if remaining <= 0:
                    break
                await sleep(remaining)
            task = asyncio.create_task(complete(index, deadline))
            pending.add(task)
            task.add_done_callback(pending.discard)
            await sleep(0)  # Start overdue arrivals too, without starving the event loop.
        if pending:
            await asyncio.gather(*tuple(pending), return_exceptions=True)
    finally:
        for task in tuple(pending):
            task.cancel()
        if pending:
            await asyncio.gather(*tuple(pending), return_exceptions=True)
    return stats
