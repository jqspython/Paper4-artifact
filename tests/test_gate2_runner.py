"""Artificial pipeline checks only; no Gate-2 trace or scientific results."""
from pathlib import Path
import tomllib
import unittest
from unittest.mock import patch

import numpy as np

import gate2_runner as runner


class RunnerChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = tomllib.loads((Path(__file__).parents[1] / 'configs/gate2.toml').read_text())

    def test_matrix_and_past_independence(self):
        choices = runner.configurations(self.config)
        self.assertEqual({len(v) for v in choices.values()}, {30})
        cases = runner.scenarios(self.config)
        self.assertEqual(len(cases), 13)
        self.assertEqual([(c['amplitude'], c['duration']) for c in cases if c['transform']=='flash'],
                         [(3., 15), (5., 60)])
        y = np.ones(5000)
        for case in cases:
            changed = runner.transform(y, 3000, 5000, case, 17)
            np.testing.assert_equal(changed[:3000], y[:3000])
            np.testing.assert_equal(changed, runner.transform(y, 3000, 5000, case, 17))

    def test_artificial_end_to_end_and_seed_subset(self):
        y = 20. + 10. * np.sin(np.arange(6000) / 30.)
        runner._data['artificial'] = dict(demand=y[None,:], unit_id=np.array(['fixture']),
                                         fit_end=1000, cal_end=4000, val_end=5000)
        nominal = runner.scenarios(self.config)[:1]
        with patch.object(runner, 'scenarios', return_value=nominal):
            rows = runner.evaluate_job(('artificial', 0, 'persistence', 0, None), self.config)
            self.assertEqual(len(rows), 300)
            data = [dict(zip(runner.FIELDS, row)) for row in rows]
            self.assertEqual(sum(r['status']=='NO_RANK' for r in data), 2)
            self.assertTrue(all(r['contract'] == 1 for r in data))
            subset = {'nominal': {method: {1} for method in runner.METHODS}}
            extra = runner.evaluate_job(('artificial', 0, 'lightgbm', 29, subset), self.config)
            self.assertEqual(len(extra), 5)
            self.assertTrue(all(dict(zip(runner.FIELDS, row))['phase']=='test' for row in extra))
        del runner._data['artificial']

    def test_hindsight_flash_keeps_zero_floor_repetitions(self):
        y = np.ones(6440)
        y[:1000] = 4.
        left = int(np.random.default_rng(17).integers(120, 1320 - 15 + 1))
        y[5000 + left + 5] = 4.
        runner._data['artificial'] = dict(demand=y[None,:], unit_id=np.array(['fixture']),
                                         fit_end=1000, cal_end=4000, val_end=5000)
        flash = [c for c in runner.scenarios(self.config) if c['name']=='flash3x15']
        with patch.object(runner, 'scenarios', return_value=flash):
            rows = runner.evaluate_job(('artificial', 0, 'persistence', 0, None), self.config)
        diagnostics = [dict(zip(runner.FIELDS, row)) for row in rows if row[7]=='hindsight']
        self.assertEqual(len(diagnostics), 5 * 3 * 30)
        self.assertEqual({r['replicate'] for r in diagnostics}, set(self.config['design']['seeds']))
        self.assertTrue(any(r['physical_floor']==0 for r in diagnostics))
        self.assertTrue(any(r['physical_floor']>0 for r in diagnostics))
        del runner._data['artificial']


if __name__ == '__main__':
    unittest.main()
