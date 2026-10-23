import math
import unittest

from paper4 import Actuator, DelayedACI, HysteresisGate, conformal_quantile


class CoreChecks(unittest.TestCase):
    def test_rank_and_delayed_issued_bound(self):
        self.assertTrue(math.isinf(conformal_quantile(range(99), .005)))
        self.assertEqual(conformal_quantile(range(199), .005), 198)
        model = DelayedACI(horizon=2, scores=range(1000))
        issued = model.issue(0, 10.)
        initial_alpha = model.alpha
        self.assertIsNone(model.reveal(0, 100000.))
        self.assertIsNone(model.reveal(1, 100000.))
        self.assertEqual(model.alpha, initial_alpha)
        self.assertEqual(model.reveal(2, issued.upper + 1), 1)
        self.assertLess(model.alpha, initial_alpha)
        self.assertEqual(issued.alpha, initial_alpha)
        with self.assertRaises(ValueError):
            model.reveal(2, 0.)
        with self.assertRaises(ValueError):
            model.issue(2, 0.)

    def test_projection_counterexample_and_certificate(self):
        model = DelayedACI(horizon=2, scores=[0.] * 1000)
        for t in range(2002):
            if t < 2000:
                model.issue(t, 0.)
            model.reveal(t, float(t + 1))
        count, gap, accounted = model.certificate()
        self.assertEqual(count, 2000)
        self.assertEqual(sum(model.errors), count)
        self.assertAlmostEqual(gap, .95)
        self.assertAlmostEqual(gap, accounted)
        self.assertGreater(model.projection_sum, 18.)

    def test_gate_hold_and_contiguous_recovery(self):
        gate = HysteresisGate(hold=5, ok=3)
        self.assertTrue(gate.step(0, .3, .2, False))
        for t in range(1, 5):
            self.assertTrue(gate.step(t, 0., .2, False))
        self.assertFalse(gate.step(5, 0., .2, False))
        self.assertTrue(gate.step(6, 0., math.inf, False))
        self.assertTrue(gate.step(7, 0., .2, True))
        for t in range(8, 12):
            self.assertTrue(gate.step(t, 0., .2, False))
        self.assertFalse(gate.step(12, 0., .2, False))

    def test_unprojected_delayed_pathwise_bound(self):
        for horizon in [1, 2, 5]:
            model = DelayedACI(gamma=.2, horizon=horizon, clip=None, scores=[0.] * 1000)
            for t in range(1000 + horizon):
                if t < 1000:
                    model.issue(t, 0.)
                matured = model.pending.get(t)
                truth = matured.upper + 1 if matured and math.isfinite(matured.upper) else 0.
                model.reveal(t, truth)
                self.assertGreaterEqual(model.alpha, -.2 * .95 * (horizon + 1) - 1e-12)
                self.assertLessEqual(model.alpha, 1 + .2 * .05 * (horizon + 1) + 1e-12)
                if model.errors:
                    n, gap, accounted = model.certificate()
                    bound = (.95 + .2 * (horizon + 1)) / (.2 * n)
                    self.assertLessEqual(abs(gap), bound + 1e-12)
                    self.assertAlmostEqual(gap, accounted)

    def test_actuation_delay_bounds_and_scale_down_window(self):
        actuator = Actuator(maximum=6, max_change=4)
        self.assertEqual(actuator.horizon, 2)
        observed = []
        for t, recommendation in enumerate([6, 6, 6, 1, 1, 1, 1, 1, 1]):
            observed.append(actuator.begin(t))
            actuator.command(t, recommendation)
        self.assertEqual(observed[:4], [10., 10., 50., 60.])
        self.assertEqual(observed[4:8], [60.] * 4)
        self.assertEqual(observed[8], 20.)
        with self.assertRaises(ValueError):
            actuator.command(8, 1)

    def test_ideal_inclusion_and_capacity_limit_exception(self):
        for upper in [.1, 9., 10., 23.5, 100.]:
            capacity = math.ceil(upper / 10.) * 10.
            for truth in [0., upper, upper + .01, 120.]:
                self.assertLessEqual(int(truth > capacity), int(truth > upper))
        upper, truth, limited_capacity = 100., 90., 40.
        self.assertFalse(truth > upper)
        self.assertTrue(truth > limited_capacity)


if __name__ == '__main__':
    unittest.main()
