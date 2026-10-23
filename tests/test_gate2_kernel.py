"""Constructed trajectories checked against the pre-Gate-2 Python mechanisms."""

import math
import unittest

import numpy as np

from cdt_pilot import BACCLite, CDT, HPA
from gate2_kernel import replay
from paper4 import Actuator


def reference(y, q, capacity, h_actual, h_assumed, maximum, rate, stable_seconds,
              kind, epsilon, parameter, cal_scores, targets):
    plant = Actuator(delay_seconds=60 * h_actual, capacity=capacity, maximum=maximum,
                     max_change=rate, stable_seconds=stable_seconds)
    if kind == 0:
        model = CDT(capacity, maximum, epsilon, parameter)
    elif kind == 1:
        model = HPA(capacity, parameter)
    elif kind == 2:
        model = BACCLite(capacity, epsilon, h_assumed, len(y), cal_scores,
                         window=720, pace_scale=parameter)
    else:
        model = None
    violated = paid_cost = ready_cost = actions = 0
    value = minimum = 0.
    rows = []
    for t in range(len(y)):
        effective = plant.begin(t)
        ready = plant.replicas
        paid = ready + sum(max(0, delta) for _, delta in plant.pending)
        loss = int(y[t] > effective)
        target, used = model.decide(t, float(y[t]), float(q[t]), loss) if model else (targets[t], 0.)
        before = ready + sum(delta for _, delta in plant.pending)
        plant.command(t, target)
        after = ready + sum(delta for _, delta in plant.pending)
        actions += int(before != after)
        if kind == 0:
            model.feedback(loss)
            value = model.value
            minimum = min(minimum, value)
        violated += loss
        paid_cost += paid
        ready_cost += ready
        rows.append((y[t], loss, ready, paid, target, used))
    return violated, paid_cost, ready_cost, actions, value, minimum, np.asarray(rows)


class KernelChecks(unittest.TestCase):
    def check_case(self, kind, lag, maximum, rate, stable, scores, length=1900,
                   assumed=2, epsilon=.05):
        t = np.arange(length)
        y = np.maximum(0., 30 + 24 * np.sin(t / 15) + (t % 97 < 7) * 110 - (t % 127 > 100) * 60)
        q = np.maximum(0., .8 * np.roll(y, 3) + 2 * np.sin(t / 7))
        targets = (t % 19).astype(float) + (t % 3) / 4
        parameter = (.25, .7, 1.5, 0.)[kind]
        arguments = (y, q, 10., lag, assumed, maximum, rate, stable, kind, epsilon, parameter, scores, targets)
        actual = replay(*arguments, record=True)
        expected = reference(*arguments)
        self.assertEqual(actual[:4], expected[:4])
        np.testing.assert_allclose(actual[4:6], expected[4:6], rtol=0, atol=1e-12)
        # Ready/paid/request/loss must agree exactly, as must issue-time alpha/lambda.
        np.testing.assert_array_equal(actual[-1], expected[-1])
        compact = replay(*arguments, record=False)
        self.assertEqual(compact[:6], actual[:6])
        self.assertEqual(compact[-1].shape, (0, 6))

    def test_each_controller_and_physical_constraints(self):
        scores = np.tile(np.array([-.1, .05, 1., 1., 1.2, 20., 25.]), 120)
        for lag, maximum, rate, stable in ((1, 1, 1, 0), (4, 8, 1, 300), (11, 16, 4, 600)):
            for kind in range(4):
                with self.subTest(kind=kind, lag=lag, maximum=maximum):
                    self.check_case(kind, lag, maximum, rate, stable, scores)

    def test_bacc_empty_and_partial_score_ring(self):
        for count in (0, 3, 719, 720):
            with self.subTest(initial_scores=count):
                self.check_case(2, 3, 16, 2, 300, np.ones(count), length=1600)

    def test_bacc_budgets_and_assumed_lags(self):
        for epsilon, assumed in zip((.01, .02, .03, .05, .07, .10), (1, 2, 3, 4, 6, 11)):
            with self.subTest(epsilon=epsilon, assumed=assumed):
                self.check_case(2, 4, 16, 1, 300, np.ones(720), length=1600,
                                assumed=assumed, epsilon=epsilon)

    def test_invalid_inputs(self):
        base = (np.ones(10), np.ones(10), 10., 2, 2, 16, 4, 300, 0, .05, .25)
        with self.assertRaises(ValueError):
            replay(*base[:8], 3, *base[9:], targets=np.zeros(3))
        with self.assertRaises(ValueError):
            replay(np.array([math.nan]), np.ones(1), *base[2:])


if __name__ == '__main__':
    unittest.main()
