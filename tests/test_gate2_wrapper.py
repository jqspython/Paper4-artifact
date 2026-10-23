"""Synthetic properties only; no traces, learned models or tuning are loaded."""

import math
import unittest

import numpy as np

from gate2_wrapper import (BACCCal, ETA_CANDIDATES, FeedbackWrapper,
                           hpa_request, pac_requests, point_margin_request,
                           wrapped_request)
from paper4 import Actuator


def plant(maximum=8, rate=1, lag=11, stable=300):
    # lag already includes the minute-end boundary: D=600s corresponds to 11.
    return Actuator(delay_seconds=60 * lag, capacity=2., maximum=maximum,
                    max_change=rate, stable_seconds=stable)


class WrapperProperties(unittest.TestCase):
    def test_integer_family_safe_threshold_and_monotonicity(self):
        for maximum in (1, 8, 16):
            for base in (0., .01, 2.5, 100., 1e300):
                values = [-maximum-10., -maximum, -.1, 0., .2, 20.]
                requests = [wrapped_request(base, v, 1, maximum) for v in values]
                self.assertEqual(requests, sorted(requests, reverse=True))
                self.assertTrue(all(isinstance(n, int) and 1 <= n <= maximum for n in requests))
                self.assertEqual(requests[:2], [maximum, maximum])
        self.assertEqual(wrapped_request(100., 10., 1, 16), 16)
        self.assertEqual(wrapped_request(16., 10., 1, 16), 6)  # Pre-clipping changes the family.
        for bad in (-.1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                wrapped_request(bad, 0., 1, 8)

    def test_old_lambda_command_then_nonprojected_feedback(self):
        wrapped = FeedbackWrapper(plant(maximum=4, lag=2), .05, 1.)
        self.assertEqual(wrapped.value, 0.)
        first = wrapped.step(0, 8., 1.)
        self.assertEqual((first['request'], first['used_lambda'], first['value']), (1, 0., -.95))
        second = wrapped.step(1, 8., 1.)
        self.assertEqual((second['request'], second['used_lambda']), (2, -.95))
        self.assertLess(wrapped.value, -1.)  # No parameter projection.
        self.assertEqual(first['paid'], first['ready'])
        for demand, base in ((math.nan, 1.), (1., -1.)):
            tick, queue, value = wrapped.plant.tick, tuple(wrapped.plant.pending), wrapped.value
            with self.assertRaises(ValueError):
                wrapped.step(2, demand, base)
            self.assertEqual((wrapped.plant.tick, tuple(wrapped.plant.pending), wrapped.value), (tick, queue, value))
        with self.assertRaises(ValueError):
            wrapped.step(3, 1., 1.)

    def test_pending_up_is_not_cancelled_and_capability_is_rejected(self):
        wrapped = FeedbackWrapper(plant(), .05, .1)
        wrapped.step(0, 0., 8.)
        pending = tuple(wrapped.plant.pending)
        self.assertTrue(pending)
        wrapped.step(1, 0., 0.)
        self.assertEqual(tuple(wrapped.plant.pending), pending)
        with self.assertRaises(ValueError):
            FeedbackWrapper(plant(), .05, .1, cancel_pending_up=True)
        altered = plant()
        altered.cancel_pending_up = True
        with self.assertRaises(ValueError):
            FeedbackWrapper(altered, .05, .1)
        class CancelActuator(Actuator):
            pass
        with self.assertRaises(ValueError):
            FeedbackWrapper(CancelActuator(), .05, .1)

    def test_safe_blocks_from_reachable_queue_and_scale_down_histories(self):
        for maximum, rate, lag, stable in ((1, 1, 11, 0), (8, 1, 11, 300),
                                            (16, 4, 11, 600), (8, 2, 2, 0)):
            sample = FeedbackWrapper(plant(maximum, rate, lag, stable), .05, .1)
            safe, settling, window = sample.constants(lag)
            histories = ([], [maximum] * 3, [maximum, 1] * 9,
                         [maximum] * (settling + 1) + [1] * 8)
            for history in histories:
                with self.subTest(maximum=maximum, rate=rate, lag=lag, history=len(history)):
                    actuator = plant(maximum, rate, lag, stable)
                    for t, request in enumerate(history):
                        actuator.begin(t)
                        actuator.command(t, request)
                    losses = []
                    for offset in range(window):
                        t = len(history) + offset
                        effective = actuator.begin(t)
                        losses.append(int(2 * maximum > effective))
                        if offset >= settling:
                            self.assertEqual(actuator.replicas, maximum)
                        before = actuator.replicas + sum(d for _, d in actuator.pending)
                        request = wrapped_request((0., .2, 100.)[offset % 3], safe, 1, maximum)
                        actuator.command(t, request)
                        after = actuator.replicas + sum(d for _, d in actuator.pending)
                        self.assertTrue(0 <= after - before <= rate)
                        self.assertTrue(1 <= actuator.replicas <= after <= maximum)
                    self.assertLessEqual(sum(losses), settling)
                    self.assertLessEqual(sum(losses) / window, .05 + 1e-12)

    def test_dynamic_lambda_lower_bound_identity_and_risk_bound(self):
        for maximum, rate, lag in ((1, 1, 11), (8, 1, 11), (16, 4, 11), (8, 2, 2)):
            for eta in ETA_CANDIDATES:
                for zero_base in (True, False):
                    wrapped = FeedbackWrapper(plant(maximum, rate, lag), .05, eta)
                    safe, settling, window = wrapped.constants(lag)
                    losses = []
                    length = max(600, 3 * window)
                    for t in range(length):
                        demand = 2. * maximum if t % 79 < 61 else 0.
                        base = 0. if zero_base else (0., 1., 100., .1)[t % 4]
                        row = wrapped.step(t, demand, base)
                        losses.append(row['loss'])
                        self.assertTrue(row['in_envelope'])
                        self.assertGreaterEqual(row['value'], safe - window * eta - 1e-9)
                        self.assertLessEqual(row['paid'], maximum)
                    rate_observed = sum(losses) / length
                    self.assertAlmostEqual(rate_observed, .05 - wrapped.value / (eta * length), places=11)
                    bound = min(1., .05 + (-safe / eta + window) / length)
                    self.assertLessEqual(rate_observed, bound + 1e-12)

    def test_outside_contract_is_visible(self):
        wrapped = FeedbackWrapper(plant(maximum=8, lag=11), .05, .1)
        with self.assertRaises(ValueError):
            wrapped.constants(2)
        self.assertFalse(wrapped.step(0, 17., 1.)['in_envelope'])

    def test_fixed_base_suggestions_and_bacc_maturity(self):
        self.assertEqual(hpa_request(14., 2., .5), 14)
        self.assertEqual(point_margin_request(5., 2., 1.5), 4)
        cal = BACCCal(2., .05, 2, 3, [1., 2., 3.])
        self.assertEqual(cal.suggest(0, 999., 10.), 16)
        cal.suggest(1, 999., 10.)
        self.assertEqual(list(cal.scores), [1., 2., 3.])
        cal.suggest(2, 21., 10.)
        self.assertEqual(list(cal.scores), [2., 3., 2.])
        self.assertEqual(cal.alpha, .05)
        self.assertFalse(any(hasattr(cal, name) for name in ('pace', 'recent', 'violations', 'feedback')))
        with self.assertRaises(ValueError):
            cal.suggest(4, 1., 1.)
        scores, pending, last = tuple(cal.scores), dict(cal.pending), cal.last
        with self.assertRaises(ValueError):
            cal.suggest(3, 1., 1e308)
        self.assertEqual((tuple(cal.scores), cal.pending, cal.last), (scores, pending, last))

    def test_pac_reuses_mature_margin_and_rejects_no_rank(self):
        from gate2_pac import margin_path, pac_rank

        y = np.arange(160, dtype=float) % 17
        q = y / 2
        start = 129
        self.assertEqual(pac_rank(128, .05), 126)
        actual = pac_requests(y, q, 2., 2, 128, .05, start=start)
        expected = np.ceil((q[start:] + margin_path(y, q, 2, 128, .05, start=start)) / 2.)
        np.testing.assert_array_equal(actual, expected)
        future_changed = y.copy()
        future_changed[start + 1:] = 1e6
        self.assertEqual(actual[0], pac_requests(future_changed, q, 2., 2, 128, .05, start=start)[0])
        with self.assertRaisesRegex(ValueError, 'No finite PAC rank'):
            pac_requests(y, q, 2., 2, 2, .01, start=start)
        with self.assertRaisesRegex(ValueError, 'Saturated PAC rank'):
            pac_requests(y, q, 2., 2, 64, .05, start=65)

    def test_four_bases_share_only_the_proposal_interface(self):
        y = np.arange(160, dtype=float) % 17
        q = y / 2
        start = 129
        pac = pac_requests(y, q, 2., 2, 128, .05, start=start)
        for base in ('HPA', 'PointMargin', 'PAC-h', 'BACC-cal'):
            wrapped = FeedbackWrapper(plant(lag=2), .05, .1)
            cal = BACCCal(2., .05, 2, 8, [1.] * 8)
            for t in range(len(y) - start):
                truth, forecast = y[start + t], q[start + t]
                suggestion = (hpa_request(truth, 2., .5) if base == 'HPA' else
                              point_margin_request(forecast, 2., 1.5) if base == 'PointMargin' else
                              pac[t] if base == 'PAC-h' else cal.suggest(t, truth, forecast))
                row = wrapped.step(t, truth, suggestion)
                self.assertTrue(row['in_envelope'])
                self.assertEqual(row['request'], wrapped_request(suggestion, row['used_lambda'], 1, 8))
            self.assertEqual(cal.alpha, .05)

    def test_bacc_cal_is_independent_of_actual_slo_history(self):
        cold, warm = plant(rate=4, lag=2), plant(rate=4, lag=2)
        for t in range(4):
            warm.begin(t)
            warm.command(t, 8)
        wrappers = [FeedbackWrapper(cold, .05, .1), FeedbackWrapper(warm, .05, .1)]
        calibrators = [BACCCal(2., .05, 2, 8, [1.] * 8) for _ in wrappers]
        histories = [[], []]
        for t in range(20):
            suggestions = [cal.suggest(t, 16., 1.) for cal in calibrators]
            self.assertEqual(suggestions[0], suggestions[1])
            self.assertEqual(tuple(calibrators[0].scores), tuple(calibrators[1].scores))
            for i, wrapped in enumerate(wrappers):
                row = wrapped.step(t if i == 0 else t + 4, 16., suggestions[i])
                histories[i].append(row['loss'])
        self.assertNotEqual(histories[0], histories[1])
        self.assertEqual([cal.alpha for cal in calibrators], [.05, .05])


if __name__ == '__main__':
    unittest.main()
