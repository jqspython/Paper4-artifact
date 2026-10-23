"""Artificial clocks and async send functions only; never generate real traffic."""

import asyncio
import random
import unittest

from sysbench.arrival import dispatch, poisson_timestamps


class ArrivalScheduleProperties(unittest.TestCase):
    def test_local_seed_determinism_global_rng_independence_and_tick_units(self):
        random.seed(10)
        before = random.getstate()
        schedule = poisson_timestamps((4., 0., 8.), seed=42, tick_seconds=5.)
        self.assertEqual(random.getstate(), before)
        random.seed(999)
        for _ in range(100):
            random.random()
        self.assertEqual(schedule, poisson_timestamps((4., 0., 8.), seed=42))
        self.assertIsInstance(schedule, tuple)
        self.assertTrue(all(a < b for a, b in zip(schedule, schedule[1:])))
        self.assertTrue(all(0 < t < 5 or 10 < t < 15 for t in schedule))
        self.assertNotEqual(schedule, poisson_timestamps((4., 0., 8.), seed=43))
        self.assertEqual(poisson_timestamps((0., 0.), seed=42), ())
        with self.assertRaises(TypeError):
            schedule[0] = 0

    def test_invalid_rate_inputs_are_rejected(self):
        for rates in ((-1.,), (float("nan"),), (float("inf"),)):
            with self.assertRaises(ValueError):
                poisson_timestamps(rates, seed=42)
        with self.assertRaises(ValueError):
            poisson_timestamps((1.,), seed=True)


class OpenLoopProperties(unittest.IsolatedAsyncioTestCase):
    def clock(self, initial=100., lateness=0.):
        state = [initial]
        async def sleeper(delay):
            state[0] += delay + (lateness if delay > 0 else 0)
            await asyncio.sleep(0)
        return lambda: state[0], sleeper

    async def test_slow_responses_never_pace_later_dispatch_and_one_schedule_is_reusable(self):
        schedule = (0., .01, .02)
        original = tuple(schedule)
        planned = []
        for unrelated_seed in (1, 999):
            random.seed(unrelated_seed)
            clock, sleeper = self.clock()
            release, all_started = asyncio.Event(), asyncio.Event()
            started, records, actual_times = [], [], {}
            async def send(index):
                started.append(index)
                actual_times[index] = clock()
                if len(started) == 3:
                    all_started.set()
                await release.wait()
                return {"artificial": index}
            replay = asyncio.create_task(dispatch(schedule, send, records.append, timeout=1.,
                                                   start_at=100., clock=clock, sleep=sleeper))
            await asyncio.wait_for(all_started.wait(), .2)
            self.assertFalse(replay.done())
            self.assertEqual(started, [0, 1, 2])
            release.set()
            summary = await replay
            self.assertEqual((summary["dispatched"], summary["ok"]), (3, 3))
            ordered = sorted(records, key=lambda r: r["index"])
            planned.append([r["planned_at"] for r in ordered])
            self.assertEqual({r["index"]: r["actual_dispatch_at"] for r in records}, actual_times)
            self.assertTrue(all(r["actual_dispatch_at"] < r["completed_at"] or
                                r["actual_dispatch_at"] == r["completed_at"] for r in ordered))
        self.assertEqual(planned[0], planned[1])
        self.assertEqual(schedule, original)

    async def test_absolute_deadlines_preserve_plan_under_late_wakes(self):
        clock, sleeper = self.clock(lateness=.003)
        records = []
        async def send(index):
            return index
        await dispatch((0., .01, .02), send, records.append, timeout=1.,
                       start_at=100., clock=clock, sleep=sleeper)
        ordered = sorted(records, key=lambda r: r["index"])
        self.assertEqual([r["planned_at"] for r in ordered], [100., 100.01, 100.02])
        self.assertTrue(all(r["actual_dispatch_at"] >= r["planned_at"] for r in ordered))
        self.assertLessEqual(ordered[2]["actual_dispatch_at"] - ordered[2]["planned_at"], .003000001)

    async def test_timeout_and_send_exception_are_visible_without_cancelling_followups(self):
        clock, sleeper = self.clock()
        records, started = [], []
        never = asyncio.Event()
        async def send(index):
            started.append(index)
            if index == 0:
                raise RuntimeError("artificial request error")
            if index == 1:
                await never.wait()
            return index
        summary = await dispatch((0., .01, .02), send, records.append, timeout=.005,
                                 start_at=100., clock=clock, sleep=sleeper)
        self.assertEqual(started, [0, 1, 2])
        self.assertEqual({r["index"]: r["status"] for r in records}, {0: "ERROR", 1: "TIMEOUT", 2: "OK"})
        self.assertEqual((summary["error"], summary["timeout"], summary["ok"]), (1, 1, 1))
        self.assertTrue(all(r["planned_at"] is not None and r["actual_dispatch_at"] is not None
                            and r["completed_at"] is not None for r in records))

    async def test_callback_errors_are_observable_and_async_callback_supported(self):
        clock, sleeper = self.clock()
        seen = []
        async def send(index):
            return index
        async def callback(record):
            await asyncio.sleep(0)
            seen.append(record["index"])
            if record["index"] == 0:
                raise ValueError("artificial sink failure")
            if record["index"] == 1:
                raise asyncio.CancelledError
        summary = await dispatch((0., .01, .02), send, callback, timeout=1., clock=clock, sleep=sleeper)
        self.assertEqual(sorted(seen), [0, 1, 2])
        self.assertEqual(summary["ok"], 3)
        self.assertEqual(summary["callback_errors"], 2)
        self.assertIn("ValueError", summary["first_callback_error"])

    async def test_invalid_schedule_or_timeout_never_starts_a_request(self):
        started = []
        async def send(index):
            started.append(index)
        for schedule, timeout in (((.2, .1), 1.), ((-.1,), 1.), ((float("nan"),), 1.), ((0.,), 0.)):
            with self.assertRaises(ValueError):
                await dispatch(schedule, send, lambda _: None, timeout=timeout)
        self.assertFalse(started)


if __name__ == "__main__":
    unittest.main()
