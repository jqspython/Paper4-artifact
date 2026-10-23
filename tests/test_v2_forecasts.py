import tempfile
import unittest
from pathlib import Path

import numpy as np

from v2_forecasts import PredictionLedger, point_losses, seasonal_features


class ForecastRetentionChecks(unittest.TestCase):
    def test_causal_target_aligned_seasons(self):
        y = np.arange(3000, dtype=float)
        original = seasonal_features(y, 2)
        changed = y.copy()
        changed[2001:] *= 20
        np.testing.assert_allclose(original[:2001], seasonal_features(changed, 2)[:2001], equal_nan=True)
        self.assertEqual(original[2000, -4], y[562])
        self.assertTrue(np.isnan(original[100, -4]))

    def test_maturity_roundtrip_and_signed_losses(self):
        ledger = PredictionLedger(2, source='synthetic', app='check', split='check',
                                  predictor='fixed', seed=0, scenario='synthetic',
                                  model_identity='fixed-values', config_identity='synthetic-check')
        ledger.issue(0, 4.)
        ledger.reveal(0, 1.)
        ledger.issue(1, 2.)
        ledger.reveal(1, 9.)
        self.assertTrue(np.isnan(ledger.rows[0][3]))
        ledger.reveal(2, 1.)
        self.assertEqual(ledger.rows[0][4], -3.)
        with self.assertRaises(ValueError):
            ledger.reveal(4, 8.)
        ledger.reveal(3, 5.)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'work') as temporary:
            path = Path(temporary) / 'predictions.npz'
            ledger.save(path)
            with np.load(path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive['target'], [2, 3])
                np.testing.assert_array_equal(archive['prediction'], [4., 2.])
                np.testing.assert_array_equal(archive['residual'], [-3., 3.])
            with self.assertRaises(FileExistsError):
                ledger.save(path)
        mae, pinball = point_losses([1., 5.], [4., 2.])
        self.assertEqual(mae, 3.)
        self.assertAlmostEqual(pinball, 1.5)


if __name__ == '__main__':
    unittest.main()
