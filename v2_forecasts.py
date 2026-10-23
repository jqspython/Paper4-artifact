"""Causal seasonal features and lossless prediction ledgers for the v2 draft.

No fitting or trace experiment is launched by this module. A forecast is issued
after observing minute t, for minute t+h. Truth is joined only when revealed.
"""

from pathlib import Path

import numpy as np

from cdt_pilot import features


def seasonal_features(demand, horizon):
    y = np.asarray(demand, dtype=float)
    if y.ndim != 1 or not np.all(np.isfinite(y)) or np.any(y < 0):
        raise ValueError('Demand must be a finite nonnegative vector')
    if not isinstance(horizon, int) or not 1 <= horizon <= 1440:
        raise ValueError('Horizon must be integer minutes in [1,1440]')
    t = np.arange(len(y))
    columns = [features(y)]
    for period in (1440, 2880):
        indices = t + horizon - period
        values = np.full(len(y), np.nan)
        valid = indices >= 0
        values[valid] = y[indices[valid]]
        columns.append(values[:, None])
    columns.extend((np.sin(2 * np.pi * (t + horizon) / 1440)[:, None],
                    np.cos(2 * np.pi * (t + horizon) / 1440)[:, None]))
    return np.column_stack(columns)


class PredictionLedger:
    """One predictor/app/scenario ledger shared by all controller candidates.

    Issue/reveal calls must be made in time order by the runner. Metadata holds
    source, app, split, predictor, seed, scenario and model/config identity.
    The ledger stores issued predictions exactly, including negative errors.
    """

    def __init__(self, horizon, **metadata):
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError('Horizon must be a positive integer')
        required = {'source', 'app', 'split', 'predictor', 'seed', 'scenario',
                    'model_identity', 'config_identity'}
        if required - metadata.keys():
            raise ValueError('Missing prediction provenance: ' + ', '.join(sorted(required - metadata.keys())))
        self.horizon, self.metadata = horizon, metadata
        self.rows, self.pending = [], {}
        self.observed_through = -1

    def issue(self, t, prediction):
        if (not isinstance(t, int) or t < 0 or
                (self.rows and t <= self.rows[-1][0]) or t < self.observed_through or
                not np.isfinite(prediction) or prediction < 0):
            raise ValueError('Issue times must increase; predictions must be finite and nonnegative')
        row = [t, t + self.horizon, float(prediction), np.nan, np.nan]
        self.pending[row[1]] = len(self.rows)
        self.rows.append(row)

    def reveal(self, t, truth):
        if (not isinstance(t, int) or t <= self.observed_through or
                (self.rows and t < self.rows[-1][0]) or
                not np.isfinite(truth) or truth < 0):
            raise ValueError('Reveal times must increase; truth must be finite and nonnegative')
        if self.pending and min(self.pending) < t:
            raise ValueError('A pending target was skipped; reveal it before advancing')
        self.observed_through = t
        index = self.pending.pop(t, None)
        if index is not None:
            row = self.rows[index]
            row[3], row[4] = float(truth), float(truth) - row[2]

    def save(self, path):
        path = Path(path)
        if path.suffix != '.npz':
            raise ValueError('Use a .npz path')
        path.parent.mkdir(parents=True, exist_ok=True)
        data = np.asarray(self.rows, dtype=float).reshape(-1, 5)
        with path.open('xb') as stream:
            np.savez_compressed(stream, issue=data[:, 0].astype('int64'),
                                target=data[:, 1].astype('int64'), prediction=data[:, 2],
                                truth=data[:, 3], residual=data[:, 4],
                                horizon=self.horizon, observed_through=self.observed_through,
                                metadata_keys=np.array(list(self.metadata), dtype=str),
                                metadata_values=np.array([str(v) for v in self.metadata.values()], dtype=str))


def point_losses(truth, prediction, tau=.9):
    y, q = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    if (y.shape != q.shape or not y.size or not 0 < tau < 1 or
            not np.all(np.isfinite(y)) or not np.all(np.isfinite(q))):
        raise ValueError('Losses need matching finite nonempty arrays and tau in (0,1)')
    residual = y - q
    return float(np.mean(np.abs(residual))), float(np.mean(np.maximum(tau * residual, (tau - 1) * residual)))
