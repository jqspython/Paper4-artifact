import math
import unittest

import numpy as np

from cdt_pilot import BACCLite, CDT, features, safety_constants, service_lag
from paper4 import Actuator


class CDTChecks(unittest.TestCase):
    def test_complete_startup_after_end_minute_command(self):
        for seconds, lag in [(0, 1), (60, 2), (180, 4), (300, 6)]:
            self.assertEqual(service_lag(seconds), lag)
            plant = Actuator(capacity=10., maximum=2, delay_seconds=60 * lag)
            for t in range(lag + 1):
                self.assertEqual(plant.begin(t), 20. if t == lag else 10.)
                plant.command(t, 2)

    def test_safe_family_settling_and_average_window(self):
        for maximum, rate, delay in [(16, 4, 3), (16, 1, 3), (8, 1, 3), (1, 1, 3)]:
            safe, settling, window = safety_constants(maximum, 1, rate, delay, .05)
            model = CDT(10., maximum, .05, .4)
            model.value = safe
            plant = Actuator(capacity=10., maximum=maximum, max_change=rate, delay_seconds=60*delay)
            losses = []
            for t in range(window):
                capacity = plant.begin(t)
                loss = int(maximum * 10 > capacity)
                request, used = model.decide(t, maximum * 10, 0., loss)
                self.assertEqual(request, maximum)
                self.assertEqual(used, safe)
                plant.command(t, request)
                losses.append(loss)
            self.assertEqual(sum(losses), settling)
            self.assertLessEqual(sum(losses) / window, .05)

    def test_command_then_real_loss_update(self):
        model = CDT(10., 16, .05, .4)
        request, used = model.decide(0, 100., 100., 1)
        self.assertEqual(request, 10)
        self.assertEqual(used, 0.)
        self.assertEqual(model.value, 0.)
        model.feedback(1)
        self.assertAlmostEqual(model.value, -.38)
        self.assertEqual(model.decide(1, 100., 100., 0)[0], 11)

    def test_features_do_not_use_future_labels(self):
        original = np.arange(200, dtype=float)
        changed = original.copy()
        changed[100:] += 10000
        np.testing.assert_equal(features(original)[:100], features(changed)[:100])

    def test_bacc_maturity_and_original_plan_labels(self):
        model = BACCLite(10., .05, 2, 10, [1.]*100)
        request, alpha = model.decide(0, 5., 10., 0)
        original = model.pending[2]
        self.assertEqual(len(model.scores), 100)
        model.decide(1, 1000., 1000., 1)
        self.assertEqual(len(model.scores), 100)
        model.decide(2, 100., 0., 1)
        self.assertEqual(len(model.scores), 101)
        self.assertAlmostEqual(model.scores[-1], 101/11)
        self.assertEqual(model.units_by_miss[1], int(100 > original[1]))


if __name__ == '__main__':
    unittest.main()
