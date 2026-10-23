"""Constructed PAC-rank, mature-window, and upstream-equivalence checks only."""

import ast
from pathlib import Path
import unittest

import numpy as np
from scipy.stats import binom

from gate2_pac import margin_grid, margin_path, pac_rank


class PACChecks(unittest.TestCase):
    def test_rank_matches_binomial_definition(self):
        for window in (10, 60, 240, 480, 960, 1440, 2880):
            for delta in (.01, .02, .03, .05, .07, .10):
                for eta in (.05, .1, .2):
                    valid = np.flatnonzero(binom.sf(np.arange(window), window, 1 - delta) <= eta)
                    expected = int(valid[0] + 1) if len(valid) else None
                    self.assertEqual(pac_rank(window, delta, eta), expected)
        self.assertEqual(pac_rank(240, .05), 234)
        self.assertEqual(pac_rank(480, .01), 479)
        self.assertIsNone(pac_rank(240, .01))

    def test_mature_scores_and_future_independence(self):
        y = np.tile(np.arange(24, dtype=float), 80)
        q = np.maximum(0., y - 3)
        h, w, start = 3, 240, 400
        result = margin_path(y, q, h, w, .05, start=start)
        r = pac_rank(w, .05)
        for t in (start, start + 1, len(y) - 1):
            scores = [max(y[j] - q[j - h], 0.) for j in range(t - w + 1, t + 1)]
            self.assertEqual(result[t - start], sorted(scores)[r - 1])
        changed = y.copy()
        changed[start + 80:] += 10000
        np.testing.assert_array_equal(result[:80], margin_path(changed, q, h, w, .05, start=start)[:80])
        self.assertTrue(np.all(margin_path(np.ones(400), np.ones(400), 2, w, .05, start=300) == 0))
        with self.assertRaises(ValueError):
            margin_path(y, q, 3, 240, .01, start=start)
        with self.assertRaises(ValueError):
            margin_path(y, q, 3, 240, .05, start=240)

    def test_fixed_official_source_parity_if_cached(self):
        path = Path(__file__).resolve().parents[1] / 'work/pac-official/analysis/audit/pac_rank_replay.py'
        if not path.exists():
            self.skipTest('Optional fixed upstream cache absent; mathematical checks still run')
        parsed = ast.parse(path.read_text(encoding='utf-8'))
        functions = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in ('pac_rank', 'margins_w')]
        namespace = {'np': np, 'binom': binom, 'ETA': .05}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
        y = np.tile(np.arange(80, dtype=float) ** 1.5, 30)
        start, window = 400, 240
        for horizon in (1, 2, 6):
            rank = pac_rank(window, .05)
            self.assertEqual(namespace['pac_rank'](window, .05), rank)
            original = namespace['margins_w'](y[None, :], start + 1, len(y) - start,
                                             horizon, window, (rank,), 0)[rank][0]
            ours = margin_path(y, y, horizon, window, .05, start=start)
            np.testing.assert_array_equal(ours, original)

    def test_grid_matches_single_paths_and_omits_invalid_candidates(self):
        y = np.tile(np.arange(80, dtype=float) ** 1.5, 30)
        q = np.maximum(0., .8 * y + np.sin(np.arange(len(y))))
        windows, deltas = (240, 480), (.01, .02, .03, .05, .07, .10)
        for horizon in (1, 2, 6):
            grid = margin_grid(y, q, horizon, windows, deltas, start=800)
            for window in windows:
                for delta in deltas:
                    rank = pac_rank(window, delta)
                    if rank is None or rank > window - 1:
                        self.assertNotIn((window, delta), grid)
                    else:
                        np.testing.assert_array_equal(grid[window, delta],
                            margin_path(y, q, horizon, window, delta, start=800))
            self.assertNotIn((240, .01), grid)

    def test_compiled_grid_ties_and_large_window(self):
        t = np.arange(7000)
        y = np.maximum(0., (t % 37).astype(float) - 20)
        q = np.maximum(0., y - (t % 3))
        for horizon in (1, 11):
            grid = margin_grid(y, q, horizon, (240, 2880), (.01, .05, .10), start=3200)
            for (window, delta), path in grid.items():
                np.testing.assert_array_equal(path,
                    margin_path(y, q, horizon, window, delta, start=3200))


if __name__ == '__main__':
    unittest.main()
