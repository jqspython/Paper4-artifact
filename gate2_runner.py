"""Frozen Gate-2 sweeps. All decisions use observed history; no ACI or kind."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime as dt
import math
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import tomllib

import numpy as np

from cdt_pilot import features, safety_constants, service_lag
from gate2_kernel import replay
from gate2_pac import margin_grid

ROOT = Path(__file__).resolve().parent
FIELDS = ('trace app predictor seed scenario replicate phase budget_mode method candidate '
          'target internal_risk violations minutes paid ready actions contract physical_floor '
          'cdt_bound mae status').split()
METHODS = ['CDT', 'HPA', 'BACC-lite', 'PAC-h', 'Point+margin']
_data = {}


def configurations(config):
    s = config['search']
    result = {}
    for name, parameters in [('CDT', s['cdt_eta']), ('BACC-lite', s['bacc_pace_scale']),
                             ('PAC-h', s['pac_window'])]:
        result[name] = [dict(risk=r, parameter=p if name != 'PAC-h' else 0.,
                             window=p if name == 'PAC-h' else s['bacc_window'], stable=300)
                        for r in s['risk_levels'] for p in parameters]
    result['HPA'] = [dict(risk=.05, parameter=p, window=0, stable=w)
                     for p in s['hpa_utilization'] for w in s['hpa_stable_seconds']]
    result['Point+margin'] = [dict(risk=.05, parameter=float(p), window=0, stable=300)
                              for p in np.linspace(s['margin_min'], s['margin_max'], s['trials_per_method'])]
    assert all(len(x) == s['trials_per_method'] for x in result.values())
    return result


def scenarios(config):
    common = dict(delay=60, maximum=16, rate=4, transform='none', amplitude=1., duration=0)
    result = []
    for delay in config['stress']['delay_seconds']:
        result.append(dict(common, name='nominal' if delay == 60 else f'delay{delay}', delay=delay))
    for transform in ['ramp', 'step']:
        for amplitude in config['stress'][transform + 's']:
            result.append(dict(common, name=f'{transform}{amplitude:g}', delay=180,
                               transform=transform, amplitude=amplitude))
    for amplitude, duration in zip(config['stress']['flash_amplitudes'], config['stress']['flash_minutes']):
        result.append(dict(common, name=f'flash{amplitude:g}x{duration}', delay=180,
                           transform='flash', amplitude=amplitude, duration=duration))
    result.extend([dict(common, name='rate1', delay=180, rate=1),
                   dict(common, name='cap8_rate1', delay=180, rate=1, maximum=8)])
    return result


def transform(demand, start, end, case, seed):
    changed = np.asarray(demand[:end], dtype=np.float64).copy()
    length = end - start
    multipliers = np.ones(length)
    if case['transform'] == 'ramp':
        multipliers = np.linspace(1., case['amplitude'], length)
    elif case['transform'] == 'step':
        multipliers[length // 2:] = case['amplitude']
    elif case['transform'] == 'flash':
        rng = np.random.default_rng(seed)
        for day in range(math.ceil(length / 1440)):
            left = day * 1440 + int(rng.integers(120, 1320 - case['duration'] + 1))
            multipliers[left:min(length, left + case['duration'])] = case['amplitude']
    changed[start:end] *= multipliers
    return changed


def model_for(demand, fit_end, config, seed):
    import lightgbm as lgb
    h = service_lag(config['plant']['assumed_startup_seconds'])
    indices = np.arange(60, fit_end - h)
    f = config['forecast']
    params = dict(objective='regression', verbosity=-1, num_threads=f['threads'],
                  deterministic=True, force_col_wise=True, seed=seed, bagging_seed=seed,
                  feature_fraction_seed=seed, bagging_fraction=f['bagging_fraction'],
                  feature_fraction=f['feature_fraction'], bagging_freq=f['bagging_freq'],
                  num_leaves=f['num_leaves'], min_data_in_leaf=20)
    return lgb.train(params, lgb.Dataset(features(demand)[indices], label=demand[indices + h]),
                     num_boost_round=f['rounds'])


def load_trace(trace):
    if trace not in _data:
        with np.load(ROOT / 'data/processed/gate2' / f'{trace}.npz', allow_pickle=False) as source:
            _data[trace] = {key: source[key] for key in source.files}
    return _data[trace]


def evaluate_job(job, config):
    check_deadline(config)
    trace, index, predictor, seed, selected = job
    data = load_trace(trace)
    demand = np.asarray(data['demand'][index], dtype=np.float64)
    app = str(data['unit_id'][index])
    fit_end, cal_end, val_end = (int(data[k]) for k in ['fit_end', 'cal_end', 'val_end'])
    capacity = float(np.max(demand[:fit_end])) / config['plant']['capacity_fit_peak_divisor']
    if capacity <= 0:
        raise ValueError('Selection contract requires positive fit peak')
    model = model_for(demand, fit_end, config, seed) if predictor == 'lightgbm' else None
    choices = configurations(config)
    h = service_lag(config['plant']['assumed_startup_seconds'])
    dmax = service_lag(config['plant']['contract_startup_seconds'])
    rows = []
    phases = [('validation', cal_end, val_end), ('test', val_end, len(demand))]
    if selected is not None:
        phases = [phases[1]]
    for case in scenarios(config):
        check_deadline(config)
        repeats = config['design']['seeds'] if case['transform'] == 'flash' else [0]
        hindsight_needed = selected is None and any(np.any(
            transform(demand, val_end, len(demand), case, rep)[val_end:] > capacity * case['maximum'])
            for rep in repeats)
        for replicate in repeats:
            for phase, start, end in phases:
                changed = transform(demand, start, end, case, replicate)
                q = np.maximum(0., model.predict(features(changed), num_threads=1)) if model else changed.copy()
                y_part, q_part = changed[start:end], q[start:end]
                length = end - start
                floor = float(np.mean(y_part > capacity * case['maximum']))
                contract = floor == 0. and service_lag(case['delay']) <= dmax
                mae = float(np.mean(np.abs(q_part[:-h] - y_part[h:])))
                truth_indices = np.arange(max(fit_end + h, start - 720), start)
                scores = np.clip((changed[truth_indices] + 1) / (q[truth_indices - h] + 1), .05, 20.)
                modes = [('causal', 0.)]
                if phase == 'test' and hindsight_needed:
                    modes.append(('hindsight', floor))
                for mode, offset in modes:
                    target = min(1., offset + config['gate']['budget'])
                    pac_deltas = [offset + r for r in config['search']['risk_levels'] if offset + r < 1]
                    margins = margin_grid(changed, q, h, config['search']['pac_window'], pac_deltas,
                                          eta=config['search']['pac_confidence_failure'], start=start)
                    for method in METHODS:
                        if mode == 'hindsight' and method in ['HPA', 'Point+margin']:
                            continue  # Identical to causal replay: these policies do not read epsilon.
                        for candidate, parameters in enumerate(choices[method]):
                            if selected is not None and candidate not in selected[case['name']].get(method, []):
                                continue
                            risk = parameters['risk'] + offset
                            status = 'OK'
                            if risk >= 1 or target >= 1:
                                status = 'NONDISCRIMINATING'
                            elif method == 'PAC-h' and (parameters['window'], risk) not in margins:
                                status = 'NO_RANK'
                            bound = None
                            output = [None] * 6
                            if status == 'OK':
                                targets = np.empty(0)
                                kind = {'CDT': 0, 'HPA': 1, 'BACC-lite': 2}.get(method, 3)
                                if method == 'PAC-h':
                                    targets = np.ceil((q_part + margins[parameters['window'], risk]) / capacity)
                                elif method == 'Point+margin':
                                    targets = np.ceil(q_part / capacity + parameters['parameter'])
                                result = replay(y_part, q_part, capacity, service_lag(case['delay']), h,
                                                case['maximum'], case['rate'], parameters['stable'], kind,
                                                risk, parameters['parameter'], scores, targets)
                                output = result[:6]
                                if method == 'CDT':
                                    assert math.isclose(result[0] / length, risk - result[4] /
                                                        (parameters['parameter'] * length), abs_tol=1e-9)
                                    if contract:
                                        safe, settling, window = safety_constants(case['maximum'], 1,
                                            case['rate'], dmax, risk)
                                        bound = min(1., risk + (-safe / parameters['parameter'] + window) / length)
                                        assert result[0] / length <= bound + 1e-9
                                        assert result[5] >= safe - window * parameters['parameter'] - 1e-8
                            violations, paid, ready, actions, _, _ = output
                            rows.append((trace, app, predictor, seed, case['name'], replicate, phase,
                                         mode, method, candidate, target, risk, violations, length,
                                         paid, ready, actions, int(contract), floor, bound, mae, status))
    return rows


def connect(config):
    db = sqlite3.connect(ROOT / 'work/gate2.sqlite')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS runs (trace TEXT,app TEXT,predictor TEXT,seed INTEGER,'
               'scenario TEXT,replicate INTEGER,phase TEXT,budget_mode TEXT,method TEXT,candidate INTEGER,'
               'target REAL,internal_risk REAL,violations INTEGER,minutes INTEGER,paid REAL,ready REAL,'
               'actions INTEGER,contract INTEGER,physical_floor REAL,cdt_bound REAL,mae REAL,status TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS jobs (job_key TEXT PRIMARY KEY)')
    db.execute('CREATE INDEX IF NOT EXISTS runs_by_app ON runs(trace,app,predictor,seed,phase,budget_mode)')
    db.execute('CREATE TABLE IF NOT EXISTS configs (method TEXT,candidate INTEGER,risk REAL,'
               'parameter REAL,window INTEGER,stable INTEGER,PRIMARY KEY(method,candidate))')
    db.executemany('INSERT OR IGNORE INTO configs VALUES (?,?,?,?,?,?)',
                   [(name, i, p['risk'], p['parameter'], p['window'], p['stable'])
                    for name, options in configurations(config).items() for i, p in enumerate(options)])
    db.commit()
    return db


def check_deadline(config):
    deadline = dt.datetime.fromisoformat(config['design']['deadline'])
    if dt.datetime.now(dt.timezone.utc) > deadline:
        raise RuntimeError('Gate-2 deadline passed; do not extend or run new trials')


def frozen_check(config):
    check_deadline(config)
    subprocess.check_output(['git', 'rev-parse', 'gate2-protocol-v1'], cwd=ROOT)
    critical = ['configs/gate2.toml', 'docs/gate2-protocol.md', 'gate2_runner.py',
                'gate2_kernel.py', 'gate2_pac.py', 'scripts/prepare_gate2_data.py',
                'cdt_pilot.py', 'paper4.py', 'scripts/analyze_gate2.py']
    subprocess.check_call(['git', 'diff', '--exit-code', 'gate2-protocol-v1', '--', *critical], cwd=ROOT,
                           stdout=subprocess.DEVNULL)


def run(stage):
    config = tomllib.loads((ROOT / 'configs/gate2.toml').read_text(encoding='utf-8'))
    frozen_check(config)
    db = connect(config)
    done = {x[0] for x in db.execute('SELECT job_key FROM jobs')}
    jobs = []
    if stage == 'primary':
        for trace in config['design']['traces']:
            assert len(load_trace(trace)['unit_id']) == config['design']['apps_per_trace']
            for index in range(config['design']['apps_per_trace']):
                for predictor in config['design']['predictors']:
                    seed = config['design']['seeds'][0] if predictor == 'lightgbm' else 0
                    key = f'primary:{trace}:{index}:{predictor}'
                    if key not in done:
                        jobs.append((key, (trace, index, predictor, seed, None)))
    else:
        sys.path.insert(0, str(ROOT / 'scripts'))
        from analyze_gate2 import selection_sets
        db.row_factory = sqlite3.Row
        for trace in config['design']['traces']:
            for index, app in enumerate(load_trace(trace)['unit_id']):
                validation = [dict(row) for row in db.execute(
                    "SELECT * FROM runs WHERE trace=? AND app=? AND predictor='lightgbm' "
                    "AND seed=17 AND phase='validation' AND budget_mode='causal'", (trace, str(app)))]
                if not validation:
                    raise RuntimeError('Primary validation missing; seeds may not start')
                selected = selection_sets(validation, config)
                for seed in config['design']['seeds'][1:]:
                    key = f'seeds:{trace}:{index}:{seed}'
                    if key not in done:
                        jobs.append((key, (trace, index, 'lightgbm', seed, selected)))
        db.row_factory = None
    started = time.monotonic()
    print('Gate2', stage, 'pending jobs', len(jobs), flush=True)
    with ProcessPoolExecutor(max_workers=config['design']['workers']) as pool:
        futures = {pool.submit(evaluate_job, job, config): key for key, job in jobs}
        for completed, future in enumerate(as_completed(futures), 1):
            key = futures[future]
            rows = future.result()
            with db:
                db.executemany('INSERT INTO runs VALUES (' + ','.join(['?'] * len(FIELDS)) + ')', rows)
                db.execute('INSERT INTO jobs VALUES (?)', (key,))
            print(f'{stage} {completed}/{len(jobs)} {key}: {len(rows)} rows; '
                  f'{time.monotonic()-started:.1f}s', flush=True)
    print('Completed', stage, 'rows', db.execute('SELECT COUNT(*) FROM runs').fetchone()[0], flush=True)
    db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['primary', 'seeds'], required=True)
    run(parser.parse_args().stage)
