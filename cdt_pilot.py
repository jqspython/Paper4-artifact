"""One-trace CDT capacity pilot. Frozen legacy ACI code is not used."""

from collections import deque
import argparse
import csv
import gzip
import math
from pathlib import Path
import tomllib

import numpy as np

from paper4 import Actuator


def service_lag(delay_seconds):
    """End of minute t request: complete startup, then serve at t+1+ceil(D/60)."""
    return 1 + math.ceil(delay_seconds / 60)


def safety_constants(maximum, minimum, rate, delay_bound, epsilon_safe):
    if not (1 <= minimum <= maximum and rate >= 1 and delay_bound >= 1 and 0 < epsilon_safe < 1):
        raise ValueError('Invalid safety contract')
    requests = math.ceil((maximum - minimum) / rate)
    settling = delay_bound + requests - 1 if requests else 0
    return -maximum, settling, max(1, math.ceil(settling / epsilon_safe))


class CDT:
    def __init__(self, capacity, maximum, epsilon, eta):
        self.capacity, self.maximum, self.epsilon, self.eta = capacity, maximum, epsilon, eta
        self.value = 0.

    def decide(self, t, demand, prediction, loss):
        requested = math.ceil(max(0., prediction) / self.capacity - self.value)
        return max(1, min(self.maximum, requested)), self.value

    def feedback(self, loss):
        self.value += self.eta * (self.epsilon - loss)


class HPA:
    def __init__(self, capacity, utilization):
        self.capacity, self.utilization = capacity, utilization

    def decide(self, t, demand, prediction, loss):
        return math.ceil(demand / (self.capacity * self.utilization)), self.utilization


class Maximum:
    def __init__(self, maximum):
        self.maximum = maximum

    def decide(self, t, demand, prediction, loss):
        return self.maximum, 0.


class BACCLite:
    """BACC v2 ratio calibration + pacing + budget ceiling, single-point windows.

    Fixed capacity replaces the measured PAVA service profile. This is a local
    mechanism subset, not a reproduction of the complete published system.
    """

    def __init__(self, capacity, epsilon, horizon, length, scores, window=720, pace_scale=1.):
        self.capacity, self.epsilon, self.horizon, self.length = capacity, epsilon, horizon, length
        self.scores = deque((float(np.clip(x, .05, 20)) for x in scores), maxlen=window)
        self.recent = deque(maxlen=1440)
        self.pending = {}
        self.pace, self.pace_scale = epsilon, pace_scale
        self.violations = self.misses = self.alpha_units = 0.
        self.units_by_miss, self.bad_by_miss = [0, 0], [0, 0]

    def decide(self, t, demand, prediction, loss):
        self.recent.append(loss)
        self.violations += loss
        if t in self.pending:
            old_prediction, old_upper, old_alpha = self.pending.pop(t)
            self.scores.append(float(np.clip((demand + 1) / (old_prediction + 1), .05, 20)))
            missed = int(demand > old_upper)
            self.units_by_miss[missed] += 1
            self.bad_by_miss[missed] += loss
            self.misses += missed
            self.alpha_units += old_alpha
        pacing_error = max(sum(self.recent) / len(self.recent), self.violations / (t + 1)) - self.epsilon
        gain = self.pace_scale * (1. if pacing_error > 0 else .2)
        self.pace = float(np.clip(self.pace - gain * pacing_error, .2 * self.epsilon, .9))
        alpha = self.pace
        remaining = self.length - t - 1
        if remaining:
            # Forecast maturity adds a boundary tick; reserve only complete startup minutes.
            reserve = loss * min(max(0, self.horizon - 1), remaining)
            affordable = float(np.clip((math.floor(self.epsilon * self.length) - self.violations - reserve)
                                       / remaining, 0., 1.))
            p0, p1 = [(bad + .5) / (units + 1) for bad, units in zip(self.bad_by_miss, self.units_by_miss)]
            if affordable < p0:
                max_miss = 0.
            elif affordable < self.epsilon and p1 > p0:
                max_miss = float(np.clip((affordable - p0) / (p1 - p0), 0., 1.))
            else:
                max_miss = 1.
            conversion = float(np.clip((self.misses + .5) / (self.alpha_units + .5), .1, 10.))
            alpha = max(.1 * self.epsilon, min(alpha, max_miss / conversion))
        ordered = sorted(self.scores)
        rank = min(len(ordered), math.ceil((len(ordered) + 1) * (1 - alpha)))
        factor = ordered[rank - 1] if ordered else 1.
        upper = max(0., (prediction + 1) * factor - 1)
        self.pending[t + self.horizon] = prediction, upper, alpha
        return math.ceil(upper / self.capacity), alpha


def features(demand):
    """Row t uses observations through t, available at the end-of-minute command."""
    y = np.asarray(demand, dtype=float)
    t = np.arange(len(y))
    columns = [y[np.maximum(0, t - lag)] for lag in [0, 1, 2, 4, 9, 29, 59]]
    total, squared = np.r_[0., np.cumsum(y)], np.r_[0., np.cumsum(y * y)]
    for window in [5, 15, 60]:
        start = np.maximum(0, t + 1 - window)
        count = t + 1 - start
        mean = (total[t + 1] - total[start]) / count
        variance = (squared[t + 1] - squared[start]) / count - mean * mean
        columns.extend([mean, np.sqrt(np.maximum(0., variance))])
    columns.extend([np.sin(2 * np.pi * t / 1440), np.cos(2 * np.pi * t / 1440),
                    (y - y[np.maximum(0, t - 5)]) / 5])
    return np.column_stack(columns)


def fit_predictor(demand, fit_end, horizon, config):
    import lightgbm as lgb
    x = features(demand)
    indices = np.arange(60, fit_end - horizon)
    params = dict(objective='regression', verbosity=-1, num_threads=2,
                  seed=config['seed'], deterministic=True, force_col_wise=True,
                  num_leaves=config['num_leaves'], min_data_in_leaf=20)
    model = lgb.train(params, lgb.Dataset(x[indices], label=demand[indices + horizon]),
                      num_boost_round=config['rounds'])
    return model


def make_controller(name, parameters, capacity, config, scores, length, maximum):
    epsilon = config['experiment']['epsilon']
    horizon = service_lag(config['experiment']['assumed_delay_seconds'])
    if name == 'CDT':
        return CDT(capacity, maximum, epsilon, parameters['eta'])
    if name == 'HPA-style':
        return HPA(capacity, parameters['utilization'])
    if name == 'BACC-lite':
        return BACCLite(capacity, epsilon, horizon, length, scores, **parameters)
    if name == 'Max-capacity':
        return Maximum(maximum)
    raise ValueError('Unknown controller; PAC-h is not defined by this implementation')


def simulate(demand, prediction, name, parameters, scenario, capacity, config, scores, ledger=None):
    experiment = config['experiment']
    plant = Actuator(step_seconds=60, delay_seconds=60 * service_lag(scenario['actual_delay_seconds']),
                     capacity=capacity, minimum=1, maximum=scenario['maximum'],
                     max_change=scenario['rate'], stable_seconds=parameters.get('stable_seconds', 300))
    controller = make_controller(name, {k: v for k, v in parameters.items() if k != 'stable_seconds'},
                                 capacity, config, scores, len(demand), scenario['maximum'])
    violated = ready_cost = provisioned_cost = actions = 0
    initial_lambda = 0.
    for t, (load, forecast) in enumerate(zip(demand, prediction)):
        effective_capacity = plant.begin(t)
        ready = plant.replicas
        # Charge replicas present during this service minute, before its end command.
        paid = ready + sum(max(0, delta) for _, delta in plant.pending)
        loss = int(load > effective_capacity)
        target, parameter = controller.decide(t, float(load), float(forecast), loss)
        projected_before = ready + sum(delta for _, delta in plant.pending)
        plant.command(t, target)
        projected_after = ready + sum(delta for _, delta in plant.pending)
        actions += int(projected_before != projected_after)
        if isinstance(controller, CDT):
            controller.feedback(loss)  # Saved request used lambda_t; update only after command.
        ready_cost += ready
        provisioned_cost += paid
        violated += loss
        if ledger is not None:
            ledger.writerow([scenario['name'], name, t, load, forecast, ready, paid,
                             target, loss, parameter])
    length = len(demand)
    delay_bound = service_lag(experiment['delay_contract_seconds'])
    safe, settling, window = safety_constants(scenario['maximum'], 1, scenario['rate'],
                                             delay_bound, experiment['epsilon'])
    ceiling_excess = int(np.sum(demand > capacity * scenario['maximum']))
    # The envelope is fixed from fitting data, not selected from test maxima.
    envelope = capacity * experiment['envelope_replicas']
    contract = (scenario['maximum'] * capacity >= envelope and float(np.max(demand)) <= envelope
                and service_lag(scenario['actual_delay_seconds']) <= delay_bound)
    bound = ''
    if isinstance(controller, CDT):
        identity = experiment['epsilon'] + (initial_lambda - controller.value) / (parameters['eta'] * length)
        assert math.isclose(identity, violated / length, abs_tol=1e-10)
        if contract:
            bound = min(1., experiment['epsilon'] + ((initial_lambda - safe) / parameters['eta'] + window) / length)
            assert violated / length <= bound + 1e-12
    return dict(scenario=scenario['name'], controller=name, minutes=length, violations=violated,
                violation_rate=violated / length, ready_replica_minutes=ready_cost,
                provisioned_replica_minutes=provisioned_cost, actions=actions,
                ceiling_excess_minutes=ceiling_excess, lambda_safe=safe, settling_L=settling,
                safe_window_K=window, contract_met=contract, cdt_upper_bound=bound,
                final_parameter=controller.value if isinstance(controller, CDT) else '')


def configurations(config):
    search = config['search']
    return {
        'CDT': [dict(eta=x) for x in search['cdt_eta']],
        'HPA-style': [dict(utilization=u, stable_seconds=s) for u in search['hpa_utilization']
                      for s in search['hpa_stable_seconds']],
        'BACC-lite': [dict(pace_scale=g, window=w) for g in search['bacc_pace_scale']
                      for w in search['bacc_window']],
    }


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config_path):
    with open(config_path, 'rb') as stream:
        config = tomllib.load(stream)
    with open(config['experiment']['trace'], encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert [int(r['minute']) for r in rows] == list(range(len(rows)))
    demand = np.array([float(r['demand']) for r in rows])
    split = np.array([r['split'] for r in rows])
    fit_end, cal_end = int(np.sum(split == 'fit')), int(np.sum(np.isin(split, ['fit', 'calibration'])))
    validation = np.flatnonzero(split == 'validation')
    test = np.flatnonzero(split == 'test')
    assert np.all(np.isfinite(demand)) and np.all(demand >= 0)
    capacity = float(np.max(demand[:fit_end])) / config['experiment']['fit_peak_replicas']
    assert capacity > 0
    horizon = service_lag(config['experiment']['assumed_delay_seconds'])
    model = fit_predictor(demand, fit_end, horizon, config['forecast'])
    original_prediction = np.maximum(0., model.predict(features(demand), num_threads=2))
    calibration = np.arange(fit_end, cal_end - horizon)
    scores = np.clip((demand[calibration + horizon] + 1) / (original_prediction[calibration] + 1), .05, 20)
    candidates = configurations(config)
    assert all(len(values) == config['search']['trials_per_method'] for values in candidates.values())
    tuning, selected = [], {}
    nominal = config['scenarios'][0]
    epsilon = config['experiment']['epsilon']
    for name, choices in candidates.items():
        results = []
        for trial, parameters in enumerate(choices):
            result = simulate(demand[validation], original_prediction[validation], name,
                              parameters, nominal, capacity, config, scores)
            tuning.append(dict(controller=name, trial=trial, parameters=repr(parameters),
                               violation_rate=result['violation_rate'],
                               provisioned_replica_minutes=result['provisioned_replica_minutes']))
            results.append(result)
        feasible = [i for i, result in enumerate(results) if result['violation_rate'] <= epsilon]
        if feasible:
            best = min(feasible, key=lambda i: results[i]['provisioned_replica_minutes'])
        else:
            best = min(range(len(results)), key=lambda i: (results[i]['violation_rate'],
                                                          results[i]['provisioned_replica_minutes']))
        selected[name] = choices[best]
    selected['Max-capacity'] = {}
    output = Path('results')
    output.mkdir(exist_ok=True)
    summary = []
    with gzip.open(output / 'cdt-ledger.csv.gz', 'wt', encoding='utf-8', newline='') as stream:
        ledger = csv.writer(stream)
        ledger.writerow(['scenario', 'controller', 'tick', 'demand', 'forecast', 'ready',
                         'paid', 'request', 'violation', 'parameter_used'])
        for scenario in config['scenarios']:
            changed = demand.copy()
            changed[test] *= scenario.get('demand_multiplier', 1.)
            forecast = np.maximum(0., model.predict(features(changed), num_threads=2))
            for name, parameters in selected.items():
                summary.append(simulate(changed[test], forecast[test], name, parameters,
                                        scenario, capacity, config, scores, ledger))
    for row in summary:
        hpa = next(r for r in summary if r['scenario'] == row['scenario'] and r['controller'] == 'HPA-style')
        row['cost_relative_hpa'] = row['provisioned_replica_minutes'] / hpa['provisioned_replica_minutes']
    write_csv(output / 'cdt-pilot.csv', summary)
    write_csv(output / 'cdt-tuning.csv', tuning)
    report(config, capacity, selected, summary)
    print('Pilot completed:', len(test), 'test minutes;', len(summary), 'controller/scenario pairs')
    print('Per-replica capacity from fit maximum:', capacity)
    print('Selected on validation only:', selected)


def report(config, capacity, selected, rows):
    lines = ['# v0.2 CDT 单 trace 压力试点', '',
             '这是公开真实 trace 驱动的容量仿真，未运行真实服务或 Kubernetes。最终闸门由用户判断。', '',
             f"目标容量违约率={100*config['experiment']['epsilon']:.2f}%；单副本容量={capacity:g} 调用/分钟（拟合段峰值/8，非实测服务容量）。",
             '每个 episode 从 1 个副本开始，全部暂态计入；成本主列含在途上调副本，另保留 Ready 副本分钟。',
             '每分钟成本在服务开始、应用到期动作后统计；分钟结束的新请求从下分钟起计入。该值是容量仿真成本代理，不是云账单。',
             '共享固定 LightGBM 点预测器，只在 fit 段训练。每法8次配置试验，仅以 nominal 验证段选参；测试不改配置。', '',
             '**PAC-h-lite：NOT PRODUCED — 原文/算法定义尚未确认；没有用其它控制器冒名替代。**', '',
             '## 选定配置', '']
    lines += [f'- {name}: `{parameters}`' for name, parameters in selected.items()]
    lines += ['', '## 压力配置', '',
              '| 场景 | 副本上限 R | 每步调幅 u | 实际启动延迟/秒 | 测试需求倍率 |',
              '|---|---:|---:|---:|---:|']
    for scenario in config['scenarios']:
        lines.append(f"| {scenario['name']} | {scenario['maximum']} | {scenario['rate']} | "
                     f"{scenario['actual_delay_seconds']} | {scenario.get('demand_multiplier', 1.)} |")
    lines += ['', '分钟t结束提交请求，完整启动D秒后，在t+1+ceil(D/60)分钟首次服务。',
              '因此名义预测h=2，证明使用的索引滞后上界d_max=4；60/180/300秒分别对应2/4/6 tick。',
              '各方法共享上限、调幅和在途队列；',
              'CDT/BACC-lite缩容稳定窗固定300秒，HPA-style在等预算候选内选择0或300秒。',
              '验证选参规则：有违约率≤5%的配置则取含在途成本最低者，否则先取违约率最低者再按成本排序。',
              '', '## 实际运行结果', '',
              '| 场景 | 控制器 | 违约率 % | 含在途成本/副本分钟 | 相对 HPA 成本 | 超容量分钟 | 契约成立 |',
              '|---|---|---:|---:|---:|---:|---|']
    for row in rows:
        lines.append(f"| {row['scenario']} | {row['controller']} | {100*row['violation_rate']:.4f} | "
                     f"{row['provisioned_replica_minutes']} | {row['cost_relative_hpa']:.4f} | "
                     f"{row['ceiling_excess_minutes']} | {row['contract_met']} |")
    lines += ['', '## 理论常数与适用范围', '',
              '族为 clip(ceil(max(q,0)/c−lambda),1,R)，lambda_safe=−R；CDT 参数不投影，更新用实际容量损失。',
              '物理启动延迟契约固定180秒，对应d_max=4；名义启动延迟60秒，对应预测h=2。需求包络在看测试前固定为16c；不裁剪原始需求。',
              'K包含整个扩容暂态的平均损失：m=ceil((R−1)/u)，L=d_max+m−1，K=ceil(L/epsilon)。', '',
              '| 场景 | lambda_safe | L | K | CDT 有限时域上界 % |', '|---|---:|---:|---:|---:|']
    for row in rows:
        if row['controller'] == 'CDT':
            bound = f"{100*row['cdt_upper_bound']:.4f}" if row['cdt_upper_bound'] != '' else '不适用'
            lines.append(f"| {row['scenario']} | {row['lambda_safe']} | {row['settling_L']} | "
                         f"{row['safe_window_K']} | {bound} |")
    lines += ['', '契约成立同时要求：上限足以覆盖预先声明的16c需求包络、实测trace未越该包络、实际延迟不超180秒。',
              'R=8的上限压力不满足该统一包络；延迟300秒超出延迟契约。表中条件式常数不在这些场景提供保证。',
              '单trace、单个确定性预测器seed；没有跨trace统计推断或置信区间。违约定义为需求超过Ready容量，不是请求延迟SLI。', '',
              '## 复现和证据', '',
              '`python cdt_pilot.py --config configs/cdt-pilot.toml`', '',
              '数据来源和选样见 [pilot-data.md](pilot-data.md)，证明见 [cdt-proof.md](cdt-proof.md)，',
              '独立审核见 [cdt-review.md](cdt-review.md)，原文/基线边界见 [cdt-related-work.md](cdt-related-work.md)。',
              '原始汇总、全部验证trial与压缩逐步账本分别位于 results/cdt-pilot.csv、cdt-tuning.csv、cdt-ledger.csv.gz。',
              'BACC-lite是v2的比值校准/pacing/ceiling机制子集，固定容量替代实测PAVA画像；不是完整BACC复现。',
              'C1/C4新增实现保持暂停，旧ACI模块没有被本试点调用。']
    Path('docs/cdt-pilot.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/cdt-pilot.toml')
    run(parser.parse_args().config)
