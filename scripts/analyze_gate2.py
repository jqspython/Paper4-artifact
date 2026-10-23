"""Frozen Gate-2 selection, application-level analysis and three figure families.

Selection is a pure validation-only API used by the runner.  Main analysis refuses
to read the result database before the protocol/configuration tag is present and
unchanged.  Flash repeats and predictor seeds never increase the application n.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import math
import sqlite3
import statistics
import subprocess
import sys
import tomllib
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("CDT", "HPA", "BACC-lite", "PAC-h", "Point+margin")
SCENARIOS = (
    "nominal", "delay120", "delay180", "delay300", "delay600", "ramp1.5",
    "ramp2", "step1.5", "step2", "flash3x15", "flash5x60", "rate1", "cap8_rate1",
)
DELAY_CASES = ("delay120", "delay180", "delay300", "delay600")


def mean(values):
    values = list(values)
    return statistics.fmean(values) if values else math.nan


def finite(value):
    return value is not None and math.isfinite(float(value))


def cutoff(config):
    return config["gate"]["budget"] + config["gate"]["budget_tolerance"]


def aggregate_rows(rows, config):
    """One application/phase/seed: mean each episode's rate and per-minute cost."""
    grouped = defaultdict(list)
    for original in rows:
        row = dict(original)
        grouped[(row["scenario"], row["method"], int(row["candidate"]))].append(row)
    result = {}
    for key, group in grouped.items():
        scenario, method, candidate = key
        expected = 5 if scenario.startswith("flash") else 1
        repetitions = [int(row["replicate"]) for row in group]
        if len(repetitions) != len(set(repetitions)):
            raise ValueError(f"Duplicate run repetitions for {key}")
        statuses = {row["status"] for row in group}
        ok = (len(group) == expected and statuses == {"OK"}
              and all(row["minutes"] > 0 and finite(row["paid"])
                      and 0 <= row["violations"] <= row["minutes"] for row in group))
        unexpected = statuses - {"OK", "NO_RANK", "NONDISCRIMINATING"}
        status = ("INCOMPLETE" if len(group) != expected else
                  "+".join(sorted(unexpected)) if unexpected else
                  "NONDISCRIMINATING" if "NONDISCRIMINATING" in statuses else
                  "NO_RANK" if "NO_RANK" in statuses else "OK" if ok else "INCOMPLETE")
        item = {
            "scenario": scenario, "method": method, "candidate": candidate,
            "internal_risk": mean(r["internal_risk"] for r in group if finite(r.get("internal_risk"))),
            "target": mean(r["target"] for r in group if finite(r.get("target"))), "status": status,
            "repetitions": len(group), "expected_repetitions": expected,
            "risk": mean(r["violations"] / r["minutes"] for r in group) if ok else math.nan,
            "cost": mean(r["paid"] / r["minutes"] for r in group) if ok else math.nan,
            "paid": mean(r["paid"] for r in group) if ok else math.nan,
            "ready": mean(r["ready"] for r in group) if ok else math.nan,
            "actions": mean(r["actions"] for r in group) if ok else math.nan,
            "minutes": mean(r["minutes"] for r in group),
            "contract": int(all(r.get("contract") == 1 for r in group)),
            "physical_floor": mean(r["physical_floor"] for r in group)
            if all(finite(r.get("physical_floor")) for r in group) else math.nan,
            "cdt_bound": mean(r["cdt_bound"] for r in group)
            if all(finite(r.get("cdt_bound")) for r in group) else math.nan,
            "mae": mean(r["mae"] for r in group)
            if all(finite(r.get("mae")) for r in group) else math.nan,
            "_diagnostic_target": mean(min(1, r["physical_floor"] + config["gate"]["budget"])
                                       for r in group if finite(r.get("physical_floor"))),
            "_diagnostic_nondiscriminating": any(finite(r.get("physical_floor"))
                                                 and r["physical_floor"] + config["gate"]["budget"] >= 1
                                                 for r in group),
        }
        result[key] = item
    return result


def choose(items, config):
    """Select only usable validation points; never use test values in this API."""
    usable = [r for r in items if r["status"] == "OK"]
    if not usable:
        return None
    budget_ok = [r for r in usable if r["risk"] <= cutoff(config) + 1e-12]
    if budget_ok:
        return min(budget_ok, key=lambda r: (r["cost"], r["candidate"]))["candidate"]
    return min(usable, key=lambda r: (r["risk"], r["cost"], r["candidate"]))["candidate"]


def _cdt_items(items, config):
    target = config["gate"]["budget"]
    risks = config["search"]["risk_levels"]
    per_risk = len(config["search"]["cdt_eta"])
    return [r for r in items if math.isclose(
        r["internal_risk"] if finite(r.get("internal_risk"))
        else risks[r["candidate"] // per_risk], target, rel_tol=0, abs_tol=1e-12)]


def matched_candidate(items, reference, config):
    if reference is None or reference["status"] != "OK" or reference["risk"] > cutoff(config) + 1e-12:
        return None
    tolerance = config["gate"]["matching_tolerance"]
    matches = [r for r in items if r["status"] == "OK"
               and r["risk"] <= cutoff(config) + 1e-12
               and abs(r["risk"] - reference["risk"]) <= tolerance + 1e-12]
    return min(matches, key=lambda r: (r["cost"], r["candidate"]))["candidate"] if matches else None


def selection_plan(rows, config):
    """Return deployment/oracle/matched indices, with one common validation rule.

    Input is all *causal validation* records of one application and one primary
    predictor seed. Missing matches remain None. The runner calls selection_sets
    to schedule additional LightGBM seeds; final analysis calls this same plan.
    """
    rows = [dict(row) for row in rows]
    if any(r.get("phase", "validation") != "validation"
           or r.get("budget_mode", "causal") != "causal" for r in rows):
        raise ValueError("Selection accepts causal validation rows only")
    for field in ("trace", "app", "predictor", "seed"):
        if len({r[field] for r in rows if field in r}) > 1:
            raise ValueError(f"Selection rows mix {field}")
    aggregates = aggregate_rows(rows, config)
    items = {(scenario, method): [r for (s, m, _), r in aggregates.items()
                                 if s == scenario and m == method]
             for scenario in SCENARIOS for method in METHODS}
    allowed = lambda scene, method: (_cdt_items(items[scene, method], config)
                                     if method == "CDT" else items[scene, method])
    deployment = {m: choose(allowed("nominal", m), config) for m in METHODS}
    cdt_nominal = aggregates.get(("nominal", "CDT", deployment["CDT"]))
    bacc_match = matched_candidate(items["nominal", "BACC-lite"], cdt_nominal, config)
    plan = {}
    for scenario in SCENARIOS:
        reference = aggregates.get((scenario, "CDT", deployment["CDT"]))
        plan[scenario] = {}
        for method in METHODS:
            plan[scenario][method] = {
                "deployment": deployment[method],
                "oracle": choose(allowed(scenario, method), config),
                "matched": (bacc_match if method == "BACC-lite" else
                            matched_candidate(items[scenario, method], reference, config)
                            if method == "HPA" else None),
            }
    return plan


def selection_sets(rows, config):
    """Runner API: dict[scenario][method] -> set of locked candidate indices."""
    return {scenario: {method: {index for index in selections.values() if index is not None}
                       for method, selections in methods.items()}
            for scenario, methods in selection_plan(rows, config).items()}


def weighted_median(values, weights=None):
    """Protocol median; exactly half mass averages the boundary and next point."""
    values = list(values)
    if not values:
        return math.inf
    weights = list(weights) if weights is not None else [1.0] * len(values)
    pairs = sorted((float(v), float(w)) for v, w in zip(values, weights) if w > 0)
    total = sum(w for _, w in pairs)
    cumulative = 0.0
    for index, (value, weight) in enumerate(pairs):
        cumulative += weight
        if cumulative >= total / 2 - 1e-12:
            if abs(cumulative - total / 2) <= 1e-12 and index + 1 < len(pairs):
                return (value + pairs[index + 1][0]) / 2
            return value
    return pairs[-1][0]


def frontier_keys(items):
    """Minimize cost and risk. Keep exact duplicates; do not interpolate."""
    good = [r for r in items if r["status"] == "OK"]
    result, best_risk = set(), math.inf
    for _, tied in itertools.groupby(sorted(good, key=lambda r: (r["cost"], r["risk"])),
                                     key=lambda r: r["cost"]):
        tied = list(tied)
        group_min = min(r["risk"] for r in tied)
        if group_min < best_risk:
            result.update((r["method"], r["candidate"]) for r in tied if r["risk"] == group_min)
        best_risk = min(best_risk, group_min)
    return result


def paired_match(left, right, config):
    return (left is not None and right is not None
            and left["status"] == right["status"] == "OK"
            and max(left["risk"], right["risk"]) <= cutoff(config) + 1e-12
            and abs(left["risk"] - right["risk"]) <= config["gate"]["matching_tolerance"] + 1e-12)


def _self_check():
    cfg = {"gate": {"budget": .05, "budget_tolerance": .001, "matching_tolerance": .001},
           "search": {"risk_levels": [.01, .02, .03, .05, .07, .1], "cdt_eta": [.01, .05, .25, .5, 1]}}
    rows = []
    for scenario in SCENARIOS:
        for method in METHODS:
            for candidate in range(30):
                for rep in range(5 if scenario.startswith("flash") else 1):
                    risk = .05 if method != "CDT" or candidate // 5 == 3 else .01
                    rows.append(dict(scenario=scenario, method=method, candidate=candidate,
                                     internal_risk=cfg["search"]["risk_levels"][candidate // 5],
                                     replicate=rep, phase="validation", budget_mode="causal", status="OK",
                                     violations=round(risk * 1000), minutes=1000, paid=(30 - candidate) * 1000,
                                     ready=1000, actions=1, contract=1, physical_floor=0, cdt_bound=1, mae=0))
    plan = selection_plan(rows, cfg)
    assert plan["nominal"]["CDT"]["deployment"] == 19
    assert plan["nominal"]["HPA"]["matched"] == 29
    assert selection_sets(rows, cfg)["delay600"]["BACC-lite"] == {29}
    assert weighted_median([1, 2]) == 1.5
    assert weighted_median([1, 2, 3], [1 / 3] * 3) == 2
    assert math.isinf(weighted_median([1, math.inf]))
    assert weighted_median([1, 2, 3], [.5, .25, .25]) == 1.5
    tied = [dict(candidate=0, cost=1, risk=.05, status="OK"),
            dict(candidate=1, cost=1, risk=.049, status="OK")]
    assert choose(tied, cfg) == 0
    assert matched_candidate(tied, tied[0], cfg) == 0
    bad = dict(rows[0], scenario="flash3x15", method="PAC-h", status="NO_RANK")
    assert aggregate_rows([bad], cfg)[("flash3x15", "PAC-h", 0)]["status"] == "INCOMPLETE"
    assert aggregate_rows([dict(bad, replicate=i) for i in range(5)], cfg)[("flash3x15", "PAC-h", 0)]["status"] == "NO_RANK"
    shifted = [dict(bad, replicate=i, status="OK", internal_risk=.05 + i * .01, target=.05 + i * .01)
               for i in range(5)]
    assert math.isclose(aggregate_rows(shifted, cfg)[("flash3x15", "PAC-h", 0)]["target"], .07)
    shifted[0]["status"] = "NONDISCRIMINATING"
    assert aggregate_rows(shifted, cfg)[("flash3x15", "PAC-h", 0)]["status"] == "NONDISCRIMINATING"
    missing = [r for r in rows if not (r["scenario"] == "flash3x15" and r["method"] == "CDT"
                                      and r["candidate"] == 19 and r["replicate"] == 4)]
    assert selection_plan(missing, cfg)["flash3x15"]["CDT"]["oracle"] == 18
    points = [dict(method="A", candidate=i, cost=c, risk=r, status="OK")
              for i, (c, r) in enumerate([(1, .1), (2, .1), (2, .05), (2, .05), (3, .07)])]
    assert frontier_keys(points) == {("A", 0), ("A", 2), ("A", 3)}
    try:
        selection_sets([dict(rows[0], phase="test")], cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("Test data reached selection")
    aggregates = aggregate_rows(rows, cfg)
    unit = summarize_unit("test-trace", "test-app", "lightgbm", 17, aggregates, aggregates, plan, cfg)
    app = application_rows(unit, cfg)[0]
    assert app["compliance"] == 1 and app["rho_bacc"] == app["rho_hpa"] == 11
    missing_unit = summarize_unit("test-trace", "test-app", "lightgbm", 29, aggregates, {}, plan, cfg, unit)
    missing_app = application_rows(missing_unit, cfg)[0]
    assert missing_app["compliance"] == 0 and missing_app["rho_bacc"] == math.inf
    print("Gate-2 pure selection checks passed (synthetic rows only).")


def require_frozen(root, config_path):
    """Check the actual registered protocol, before opening any controller output."""
    for relative in ("docs/gate2-protocol.md", str(config_path.relative_to(root)).replace("\\", "/"),
                     "scripts/analyze_gate2.py"):
        frozen = subprocess.run(["git", "show", f"gate2-protocol-v1:{relative}"], cwd=root,
                                capture_output=True, check=False)
        if frozen.returncode:
            raise RuntimeError("gate2-protocol-v1 is absent or lacks " + relative)
        current = (root / relative).read_bytes()
        if frozen.stdout.replace(b"\r\n", b"\n") != current.replace(b"\r\n", b"\n"):
            raise RuntimeError("Frozen protocol/config differs from working file: " + relative)


def _stat(aggregates, scene, method, candidate):
    return aggregates.get((scene, method, candidate)) if candidate is not None else None


def summarize_unit(trace, app, predictor, seed, validation, tests, plan, config, contract_reference=None):
    cases = {}
    for scene in SCENARIOS:
        observed = [r for (s, _, _), r in tests.items() if s == scene and r["status"] == "OK"]
        contracts = {r["contract"] for r in observed}
        if len(contracts) > 1:
            raise ValueError(f"Method-dependent contract denominator: {trace}/{app}/{scene}")
        case = {"contract": contracts == {1},
                "physical_floor": observed[0]["physical_floor"] if observed else math.nan,
                "methods": {}, "ratios": {}, "match_reason": {}}
        if contract_reference is not None:
            reference = contract_reference["cases"][scene]
            if observed and case["contract"] != reference["contract"]:
                raise ValueError("Predictor/seed changed the shared contract denominator")
            case["contract"] = reference["contract"]
            case["physical_floor"] = reference["physical_floor"]
        for method in METHODS:
            case["methods"][method] = {
                mode: _stat(tests, scene, method, index)
                for mode, index in plan[scene][method].items()}
        cdt_index = plan[scene]["CDT"]["deployment"]
        cdt_val = _stat(validation, scene, "CDT", cdt_index)
        cdt_test = _stat(tests, scene, "CDT", cdt_index)
        for baseline in ("BACC-lite", "HPA"):
            index = plan[scene][baseline]["matched"]
            base_val = _stat(validation, scene, baseline, index)
            base_test = _stat(tests, scene, baseline, index)
            reason = ("NO_VALIDATION_MATCH" if index is None else
                      "VALIDATION_RISK_MISMATCH" if not paired_match(cdt_val, base_val, config) else
                      "TEST_RISK_MISMATCH_OR_MISSING" if not paired_match(cdt_test, base_test, config) else
                      "OK")
            case["match_reason"][baseline] = reason
            case["ratios"][baseline] = (cdt_test["cost"] / base_test["cost"]
                                        if reason == "OK" and base_test["cost"] > 0 else math.inf)
        cases[scene] = case
    return dict(trace=trace, app=app, predictor=predictor, seed=seed, cases=cases, plan=plan)


def application_rows(unit, config):
    cases = unit["cases"]
    all_cases = [case for case in cases.values() if case["contract"]]
    pressure = [case for scene, case in cases.items() if scene != "nominal" and case["contract"]]
    rows = []
    for method in METHODS:
        deployed = [case["methods"][method]["deployment"] for case in pressure]
        oracles = [case["methods"][method]["oracle"] for case in pressure]
        row = {field: unit[field] for field in ("trace", "app", "predictor", "seed")}
        row.update(method=method, contract_pressure_cases=len(pressure), contract_all_cases=len(all_cases),
                   compliance=mean(r is not None and r["status"] == "OK"
                                   and r["risk"] <= cutoff(config) + 1e-12 for r in deployed),
                   strict_compliance=mean(r is not None and r["status"] == "OK"
                                          and r["risk"] <= config["gate"]["budget"] + 1e-12
                                          for r in deployed),
                   oracle_compliance=mean(r is not None and r["status"] == "OK"
                                          and r["risk"] <= cutoff(config) + 1e-12 for r in oracles),
                   mean_risk=mean(r["risk"] for r in deployed)
                   if all(r is not None and r["status"] == "OK" for r in deployed) else math.nan,
                   mean_cost=mean(r["cost"] for r in deployed)
                   if all(r is not None and r["status"] == "OK" for r in deployed) else math.nan)
        nominal = cases["nominal"]["methods"][method]["deployment"]
        row["nominal_mae"] = nominal["mae"] if nominal else math.nan
        for baseline, suffix in (("BACC-lite", "bacc"), ("HPA", "hpa")):
            ratios = [case["ratios"][baseline] for case in all_cases]
            row["rho_" + suffix] = weighted_median(ratios) if method == "CDT" else math.nan
            row["matched_" + suffix] = (mean(math.isfinite(v) for v in ratios) if ratios else 0.0) if method == "CDT" else math.nan
        rows.append(row)
    return rows


def group_by_trace(rows, field, transform=None):
    grouped = defaultdict(list)
    for row in rows:
        value = row[field]
        if value is None or math.isnan(value):
            continue
        grouped[row["trace"]].append(transform(value) if transform else value)
    return dict(grouped)


def macro_value(groups, kind="mean"):
    if not groups:
        return math.nan
    if kind == "mean":
        return mean(mean(values) for values in groups.values())
    values = [v for group in groups.values() for v in group]
    weights = [1 / (len(groups) * len(group)) for group in groups.values() for _ in group]
    return weighted_median(values, weights)


def hierarchical_interval(groups, config, kind="mean"):
    """Trace then application percentile bootstrap; only three source clusters."""
    if not groups or any(not group for group in groups.values()):
        return math.nan, math.nan
    import random

    rng = random.Random(config["statistics"]["bootstrap_seed"])
    traces = list(groups)
    samples = []
    for _ in range(config["statistics"]["bootstrap_repetitions"]):
        drawn = {}
        for slot in range(len(traces)):
            values = groups[rng.choice(traces)]
            drawn[slot] = [rng.choice(values) for _ in values]
        samples.append(macro_value(drawn, kind))
    samples.sort()
    alpha = config["statistics"]["alpha"]
    # Nearest-rank percentiles retain infinity; interpolating infinity can yield NaN.
    quantile = lambda p: samples[max(0, min(len(samples) - 1, math.ceil(p * len(samples)) - 1))]
    return quantile(alpha / 2), quantile(1 - alpha / 2)


def build_summary(app_rows, units, config):
    summary, decisions = [], {}
    primary = [("lightgbm", config["design"]["seeds"][0]), ("persistence", 0)]
    conditions = primary + [("lightgbm", s) for s in config["design"]["seeds"][1:]]
    for predictor, seed in conditions:
        subset = [r for r in app_rows if r["predictor"] == predictor and r["seed"] == seed]
        points = {}
        for method in METHODS:
            selected = [r for r in subset if r["method"] == method]
            groups = group_by_trace(selected, "compliance")
            estimate = macro_value(groups)
            low, high = hierarchical_interval(groups, config)
            points[method] = estimate
            summary.append(dict(predictor=predictor, seed=seed, scope="overall", method=method,
                                metric="A_compliance", estimate=estimate, ci_low=low, ci_high=high,
                                applications=sum(map(len, groups.values())), trace_clusters=len(groups)))
            for trace, vals in groups.items():
                summary.append(dict(predictor=predictor, seed=seed, scope=trace, method=method,
                                    metric="A_compliance", estimate=mean(vals), ci_low=math.nan,
                                    ci_high=math.nan, applications=len(vals), trace_clusters=1))
            strict = macro_value(group_by_trace(selected, "strict_compliance"))
            summary.append(dict(predictor=predictor, seed=seed, scope="overall", method=method,
                                metric="strict_compliance", estimate=strict))
            oracle = macro_value(group_by_trace(selected, "oracle_compliance"))
            summary.append(dict(predictor=predictor, seed=seed, scope="overall", method=method,
                                metric="oracle_compliance", estimate=oracle))
        cdt_rows = [r for r in subset if r["method"] == "CDT"]
        hpa_by_app = {(r["trace"], r["app"]): r for r in subset if r["method"] == "HPA"}
        diffs = [dict(r, difference=r["compliance"] - hpa_by_app[r["trace"], r["app"]]["compliance"])
                 for r in cdt_rows]
        diff_groups = group_by_trace(diffs, "difference")
        low, high = hierarchical_interval(diff_groups, config)
        difference = macro_value(diff_groups)
        summary.append(dict(predictor=predictor, seed=seed, scope="overall", method="CDT-HPA",
                            metric="A_compliance_difference", estimate=difference, ci_low=low, ci_high=high))
        for field, metric, transform, kind in (
            ("rho_bacc", "B_bacc_median_ratio", None, "median"),
            ("rho_hpa", "B_hpa_success", lambda v: float(v <= config["gate"]["b_oracle_hpa_ratio"]), "mean"),
            ("matched_bacc", "B_bacc_valid_match", None, "mean"),
            ("matched_hpa", "B_hpa_valid_match", None, "mean"),
        ):
            groups = group_by_trace(cdt_rows, field, transform)
            estimate = macro_value(groups, kind)
            low, high = hierarchical_interval(groups, config, kind)
            points[metric] = estimate
            summary.append(dict(predictor=predictor, seed=seed, scope="overall", method="CDT",
                                metric=metric, estimate=estimate, ci_low=low, ci_high=high,
                                applications=sum(map(len, groups.values())), trace_clusters=len(groups)))
            for trace, vals in groups.items():
                value = weighted_median(vals) if kind == "median" else mean(vals)
                summary.append(dict(predictor=predictor, seed=seed, scope=trace, method="CDT",
                                    metric=metric, estimate=value, applications=len(vals), trace_clusters=1))
        eligible = {}
        for trace in config["design"]["traces"]:
            selected_units = [u for u in units if u["trace"] == trace and u["predictor"] == predictor and u["seed"] == seed]
            for scene in DELAY_CASES:
                eligible[trace, scene] = sum(u["cases"][scene]["contract"] for u in selected_units)
        enough = all(n >= config["gate"]["minimum_contract_apps_per_trace"] for n in eligible.values())
        decisions[predictor, seed] = dict(
            A=(enough and difference >= config["gate"]["a_hpa_improvement"] - 1e-12
               and points["CDT"] >= points["PAC-h"] - 1e-12
               and points["CDT"] >= points["BACC-lite"] - 1e-12),
            B=(enough and points["B_bacc_median_ratio"] <= config["gate"]["b_bacc_median_ratio"] + 1e-12
               and points["B_hpa_success"] >= config["gate"]["b_oracle_hpa_success_fraction"] - 1e-12),
            enough=enough, contract_counts=eligible, difference=difference, points=points)
    return summary, decisions


def paired_statistics(app_rows, units, config):
    """Auxiliary paired tests use applications, not their scenarios/minutes/seeds."""
    import scipy
    from scipy.stats import wilcoxon

    output = []
    for predictor, seed in (("lightgbm", config["design"]["seeds"][0]), ("persistence", 0)):
        for trace in config["design"]["traces"]:
            sample = [r for r in app_rows if r["trace"] == trace and r["predictor"] == predictor and r["seed"] == seed]
            lookup = {(r["app"], r["method"]): r for r in sample}
            apps = sorted({r["app"] for r in sample})
            for baseline in METHODS[1:]:
                for metric in ("compliance", "mean_risk", "mean_cost"):
                    differences = [lookup[app, "CDT"][metric] - lookup[app, baseline][metric]
                                   for app in apps if finite(lookup[app, "CDT"][metric])
                                   and finite(lookup[app, baseline][metric])]
                    if not differences:
                        statistic, p = math.nan, math.nan
                    elif all(abs(d) <= 1e-15 for d in differences):
                        statistic, p = 0.0, 1.0
                    else:
                        test = wilcoxon(differences, zero_method="pratt", alternative="two-sided", method="approx")
                        statistic, p = float(test.statistic), float(test.pvalue)
                    output.append(dict(trace=trace, predictor=predictor, seed=seed, baseline=baseline,
                                       metric=metric, n_paired_applications=len(differences),
                                       missing_pairs=len(apps) - len(differences), difference=mean(differences),
                                       wins=sum(d > 0 for d in differences), ties=sum(d == 0 for d in differences),
                                       losses=sum(d < 0 for d in differences), statistic=statistic,
                                       p_raw=p, p_holm=math.nan, scipy_version=scipy.__version__))
                counts = Counter()
                for unit in units:
                    if (unit["trace"], unit["predictor"], unit["seed"]) != (trace, predictor, seed):
                        continue
                    for scene, case in unit["cases"].items():
                        if scene == "nominal" or not case["contract"]:
                            continue
                        passed = []
                        for method in ("CDT", baseline):
                            row = case["methods"][method]["deployment"]
                            passed.append(bool(row and row["status"] == "OK" and row["risk"] <= cutoff(config) + 1e-12))
                        counts[tuple(passed)] += 1
                output.append(dict(trace=trace, predictor=predictor, seed=seed, baseline=baseline,
                                   metric="paired_budget_case_counts", both_pass=counts[True, True],
                                   cdt_only_pass=counts[True, False], baseline_only_pass=counts[False, True],
                                   both_fail=counts[False, False], p_raw=math.nan, p_holm=math.nan))
    # One conservative family across all predictors, source traces and endpoints.
    ordered = sorted((r for r in output if finite(r.get("p_raw"))), key=lambda r: r["p_raw"])
    previous = 0.0
    for rank, row in enumerate(ordered):
        previous = max(previous, min(1.0, (len(ordered) - rank) * row["p_raw"]))
        row["p_holm"] = previous
    return output


def csv_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return "NA" if math.isnan(value) else "inf" if value == math.inf else "-inf" if value == -math.inf else value
    return value


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_value(row.get(k)) for k in fields})


def make_figures(points, app_rows, summary, config, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {"CDT": "#0072B2", "HPA": "#D55E00", "BACC-lite": "#009E73",
              "PAC-h": "#CC79A7", "Point+margin": "#555555"}
    markers = dict(zip(METHODS, ("o", "s", "^", "D", "x")))
    primary_seed = config["design"]["seeds"][0]
    primary = [r for r in app_rows if r["predictor"] == "lightgbm" and r["seed"] == primary_seed]
    lookup = {(r["predictor"], r["seed"], r["method"], r["metric"]): r
              for r in summary if r["scope"] == "overall"}

    def export(fig, stem):
        for suffix in ("png", "pdf"):
            fig.savefig(folder / f"{stem}.{suffix}", dpi=300, facecolor="white")
        plt.close(fig)

    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "font.family": "DejaVu Sans"}):
        fig, axes = plt.subplots(4, 4, figsize=(17, 15), layout="constrained")
        for axis, scene in zip(axes.flat, SCENARIOS):
            selected = [p for p in points if p[0] == scene]
            for method in METHODS:
                values = [p for p in selected if p[1] == method]
                if not values:
                    continue
                axis.scatter([p[2] for p in values], [100 * p[3] for p in values],
                             s=4, alpha=.08, c=colors[method], marker=markers[method],
                             rasterized=True, linewidths=.2)
                front = [p for p in values if p[4]]
                axis.scatter([p[2] for p in front], [100 * p[3] for p in front],
                             s=11, alpha=.55, c=colors[method], marker=markers[method],
                             rasterized=True, linewidths=.4, label=method)
            axis.set_xscale("log")
            axis.set_ylim(0, 100)
            axis.axhspan(100 * cutoff(config), 100, color="#777777", alpha=.06)
            axis.axhline(100 * cutoff(config), color="black", linestyle="--", linewidth=.8)
            axis.set(title=scene, xlabel="Paid cost / locked HPA cost (log10)", ylabel="Violation minutes (%)")
        for axis in list(axes.flat)[len(SCENARIOS):]:
            axis.axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=5)
        fig.suptitle("Descriptive test candidate frontiers: one point = one application/configuration\n"
                     "LightGBM seed 17; faint = all usable points, solid = application-specific cross-method frontier; no test selection")
        export(fig, "gate2-frontiers")

        fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
        scopes = list(config["design"]["traces"]) + ["overall"]
        for axis, (predictor, seed) in zip(axes[0], (("lightgbm", primary_seed), ("persistence", 0))):
            for offset, method in enumerate(METHODS):
                values = [next((r["estimate"] for r in summary if r["predictor"] == predictor and r["seed"] == seed
                                and r["scope"] == scope and r["method"] == method and r["metric"] == "A_compliance"), math.nan)
                          for scope in scopes]
                axis.bar(np.arange(4) + (offset - 2) * .15, np.array(values) * 100,
                         width=.14, color=colors[method], label=method, hatch="/" if method == "PAC-h" else None)
            axis.set_ylim(0, 100)
            axis.set_xticks(range(4), ["Huawei private", "Huawei public", "Azure 2019", "Macro"])
            axis.tick_params(axis="x", rotation=15)
            axis.set(title=predictor + ": locked deployment", ylabel="Application × stress-case compliance (%)")
        cdt = [r for r in primary if r["method"] == "CDT"]
        for axis, suffix, title, threshold in ((axes[1, 0], "bacc", "CDT / nominal-matched BACC-lite", .9),
                                               (axes[1, 1], "hpa", "CDT / scenario-matched oracle HPA", 1.1)):
            groups = group_by_trace(cdt, "rho_" + suffix)
            pairs = sorted((value, 1 / (len(groups) * len(vals))) for vals in groups.values() for value in vals)
            x, y, cumulative = [], [], 0.0
            for value, weight in pairs:
                cumulative += weight
                if math.isfinite(value):
                    x.append(value)
                    y.append(cumulative)
            if x:
                axis.step(x, y, where="post", color=colors["CDT"])
            missing = sum(weight for value, weight in pairs if not math.isfinite(value))
            axis.text(.03, .9, f"Unmatched/invalid mass at +inf: {100 * missing:.1f}%", transform=axis.transAxes)
            axis.axvline(threshold, color="black", linestyle="--", linewidth=.8)
            axis.set(xlabel="Per-application median paid-cost ratio", ylabel="Trace-equal cumulative application mass",
                     title=title, ylim=(0, 1))
            axis.set_xscale("log")
        axes[0, 0].legend(fontsize=8, ncol=2)
        fig.suptitle("Application-level compliance and matched-cost distributions\n"
                     "Flash repeats averaged first; each source trace has total weight 1/3; failed matches stay at +infinity")
        export(fig, "gate2-compliance-cost")

        fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
        seeds = config["design"]["seeds"]
        for method in METHODS:
            values = [lookup.get(("lightgbm", seed, method, "A_compliance"), {}).get("estimate", math.nan) for seed in seeds]
            axes[0].plot(range(len(seeds)), np.array(values) * 100, label=method, color=colors[method],
                         marker=markers[method], linewidth=1)
        axes[0].set(ylabel="Macro compliance (%)", ylim=(0, 100), title="Locked deployment across predictor seeds")
        axes[0].legend(fontsize=8)
        ratios = [lookup.get(("lightgbm", seed, "CDT", "B_bacc_median_ratio"), {}).get("estimate", math.nan) for seed in seeds]
        finite_ratios = [r for r in ratios if finite(r)]
        top = max([1.2] + finite_ratios) * 1.1
        axes[1].plot(range(len(seeds)), [r if finite(r) else math.nan for r in ratios], marker="o", color=colors["CDT"])
        for i, ratio in enumerate(ratios):
            if ratio == math.inf:
                axes[1].annotate("+inf", (i, top * .94), ha="center", va="top")
                axes[1].scatter([i], [top * .85], marker="^", color="black")
        axes[1].axhline(.9, color="black", linestyle="--", linewidth=.8)
        axes[1].set(title="CDT / BACC-lite application ratio", ylabel="Trace-equal weighted median", ylim=(0, top))
        success = [lookup.get(("lightgbm", seed, "CDT", "B_hpa_success"), {}).get("estimate", math.nan) for seed in seeds]
        axes[2].plot(range(len(seeds)), np.array(success) * 100, marker="o", color=colors["CDT"])
        axes[2].axhline(70, color="black", linestyle="--", linewidth=.8)
        axes[2].set(title="Cost within 1.10 × matched oracle HPA", ylabel="Application macro fraction (%)", ylim=(0, 100))
        for axis in axes:
            axis.set_xticks(range(len(seeds)), [str(seed) for seed in seeds])
            axis.set_xlabel("LightGBM predictor seed (repeated measure, not independent n)")
        fig.suptitle("E5: deployment and matching configurations frozen using seed 17 validation\n"
                     "Other seeds replay the same candidate indices; no parameter reselection")
        export(fig, "gate2-seed-sensitivity")


def report_text(config, summary, decisions, units, slots, incomplete, hindsight, output_paths):
    def number(value, percent=False):
        if value is None or math.isnan(value):
            return "NA"
        if value == math.inf:
            return "+∞"
        return f"{100 * value:.2f}%" if percent else f"{value:.4f}"

    seed = config["design"]["seeds"][0]
    primary = decisions["lightgbm", seed]
    a_pass = all(decisions[key]["A"] for key in (("lightgbm", seed), ("persistence", 0)))
    b_pass = primary["B"]
    enough = all(decisions[key]["enough"] for key in (("lightgbm", seed), ("persistence", 0)))
    status = "INCOMPLETE" if incomplete else "证据不足" if not enough else "A/B 均达到预注册线" if a_pass and b_pass else "未同时达到 A/B 预注册线"
    gate_readout = (f"A/B 中间计算（非验收结论）：A={a_pass}；B={b_pass}。" if incomplete
                    else f"A={a_pass}；B={b_pass}。")
    lines = ["# Gate-2 决定性实验结果", "", f"执行侧计算状态：**{status}**。{gate_readout}最终闸门由用户裁定。",
             "", "这是冻结协议的实验分析报告；kind、ACI 新增实现与论文写作保持暂停。所有部署、场景 oracle 与风险匹配点仅由 validation 选择。完整 test 前沿用于描述，没有 test 择点。",
             "", "## A：应用×压力场景合规比例", "",
             "Flash 的 5 次注入先在应用×场景内平均；每个应用对其契约内压力场景等权，每份 trace 先平均应用，再对三份 trace 等权。nominal 不进入 A。",
             "", "| 预测器 | 方法 | 宏平均 | 层级 bootstrap 95% 区间 | A 有效应用数 |", "|---|---|---:|---|---:|"]
    for row in summary:
        if row["scope"] == "overall" and row["metric"] == "A_compliance" and (row["predictor"], row["seed"]) in (("lightgbm", seed), ("persistence", 0)):
            lines.append(f"| {row['predictor']} | {row['method']} | {number(row['estimate'], True)} | [{number(row.get('ci_low'), True)}, {number(row.get('ci_high'), True)}] | {row['applications']} |")
    for key in (("lightgbm", seed), ("persistence", 0)):
        decision = decisions[key]
        lines += ["", f"{key[0]}：CDT−nominal 调参 HPA={number(decision['difference'], True)}；要求至少 15 个百分点，并且 CDT 不低于 PAC-h 和 BACC-lite；该预测器 A={decision['A']}。"]
        oracle = next(r["estimate"] for r in summary if (r["predictor"], r["seed"], r["scope"], r["method"], r["metric"])
                      == (key[0], key[1], "overall", "HPA", "oracle_compliance"))
        lines.append(f"场景 validation 各自重选的普通 oracle HPA 合规宏平均={number(oracle, True)}，只作补充，不取代 A 主 HPA 对照。")
    lines += ["", "## B：锁点后匹配成本", "",
              "未匹配、未双预算内或风险差超过 0.1 个百分点的 app×case 以 +∞ 占原位置；无契约 case 的应用也保留 +∞。先应用内中位数，再按 trace 等权求总体加权中位数；累计权重恰好 0.5 时取该点与下一点的平均。",
              "", "| 主 LightGBM seed 17 指标 | 点估计 | 95% 区间 | 判据 |", "|---|---:|---|---|"]
    labels = {"B_bacc_median_ratio": ("CDT/BACC-lite 应用成本比加权中位数", "≤0.90", False),
              "B_hpa_success": ("CDT/场景 oracle HPA 应用成本比≤1.10 的宏平均比例", "≥70%", True),
              "B_bacc_valid_match": ("BACC-lite 有效匹配 case 比例", "完整报告，不删失败项", True),
              "B_hpa_valid_match": ("HPA 有效匹配 case 比例", "完整报告，不删失败项", True)}
    for row in summary:
        if row["scope"] == "overall" and row["predictor"] == "lightgbm" and row["seed"] == seed and row["metric"] in labels:
            label, threshold, percent = labels[row["metric"]]
            lines.append(f"| {label} | {number(row['estimate'], percent)} | [{number(row.get('ci_low'), percent)}, {number(row.get('ci_high'), percent)}] | {threshold} |")
    lines += ["", "## 契约覆盖、候选与数据完整性", "",
              "| 来源 | delay120 | delay180 | delay300 | delay600 |", "|---|---:|---:|---:|---:|"]
    for trace in config["design"]["traces"]:
        lines.append("| " + trace + " | " + " | ".join(str(primary["contract_counts"][trace, case]) for case in DELAY_CASES) + " |")
    lines += ["", "每份来源的上述四个延迟 case 均需至少 30 个契约内应用，否则不能通过相应判据。Huawei 两份来源共享厂商背景，三份 trace 不等于三个独立云厂商。",
              "", "| 方法 | 每应用候选槽 | nominal validation 可运行槽最小/最大 |", "|---|---:|---|"]
    for method in METHODS:
        counts = slots[method]
        lines.append(f"| {method} | 30 | {min(counts) if counts else 0}/{max(counts) if counts else 0} |")
    lines += ["", "PAC-h 的不可行二项秩槽保留为 NO_RANK；30 个预算槽和实际可运行数分开计算，不把无穷端点裁成有限值。",
              "", f"缺失或意外失败的必需记录：{len(incomplete)}。全部具体原因记录在本报告末尾，存在缺失时 A/B 点估计仅作中间诊断，状态为 INCOMPLETE。",
              "", "## E5：预测器种子敏感性", "",
              "| LightGBM seed | CDT 合规宏平均 | HPA 合规宏平均 | PAC-h 合规宏平均 | BACC-lite 合规宏平均 | CDT/BACC 中位比 | HPA 成本成功比例 |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for s in config["design"]["seeds"]:
        p = decisions["lightgbm", s]["points"]
        lines.append(f"| {s} | {number(p['CDT'], True)} | {number(p['HPA'], True)} | {number(p['PAC-h'], True)} | {number(p['BACC-lite'], True)} | {number(p['B_bacc_median_ratio'])} | {number(p['B_hpa_success'], True)} |")
    lines += ["", "这些是同一应用的重复测量；另外四个种子继承 seed 17 validation 已锁配置，不增加独立 n、不重调参数。persistence 使用 seed 0 的同等 30 槽扫描，详见汇总表。",
              "", "## E4：事后可行性诊断", "",
              "只对 test 物理下限大于零的 case 使用 hindsight budget；预算及内部风险读入 test 物理下限，属于事后信息。表中 deployment 行沿用 causal validation 索引，HPA/固定余量从 causal 同索引重放关联；其余候选只作描述。NONDISCRIMINATING 行不用于成本优劣，不进入 A/B。",
              "", f"hindsight/复用候选记录共 {len(hindsight)} 行，其中已锁 deployment 对应记录 {sum(r['locked_deployment'] for r in hindsight)} 行。风险、物理下限、超下限风险、成本及状态见压缩 CSV；没有事后选择最优候选。",
              "", "| 预测器 | 方法 | 已锁点 OK/全部诊断 case | 应用内平均超物理下限风险的 trace 宏平均 |", "|---|---|---:|---:|"]
    for predictor in ("lightgbm", "persistence"):
        for method in METHODS:
            selected = [r for r in hindsight if r["predictor"] == predictor and r["method"] == method and r["locked_deployment"]]
            grouped = defaultdict(list)
            for row in selected:
                if row["status"] == "OK":
                    grouped[row["trace"], row["app"]].append(row["excess_risk"])
            app_means = [dict(trace=trace, excess=mean(values)) for (trace, _), values in grouped.items()]
            excess = macro_value(group_by_trace(app_means, "excess"))
            lines.append(f"| {predictor} | {method} | {sum(r['status'] == 'OK' for r in selected)}/{len(selected)} | {number(excess, True)} |")
    lines += ["", "以上 E4 均值只概括状态 OK 的事后诊断行，分母和无效数明确保留；不用于 A/B 或方法总体优劣判定。",
              "", "## 统计与图注", "",
              "独立分析单位为应用；先 trace、再应用重采样的 2000 次 percentile bootstrap，seed=20261018。总体只有 3 个来源簇，区间稳定性和簇级外推能力有限。每 trace 的配对 Wilcoxon 使用应用内压力聚合值、Pratt 零差处理和双侧正态近似；Holm 在全部主预测器/来源/辅助终点比较组成的一个家族内校正。另报同 case 双预算配对计数；p 值不参与 A/B。",
              "", "前沿图：每点为应用×配置，flash 已均值；淡点为全部可用候选，实点为该应用对应场景的跨方法非支配点。成本除以该应用场景中 nominal-validation 锁定 HPA 的 paid 成本，横轴 log10；不同应用不连成可部署前沿。纵轴为实际容量违约分钟比例，灰区高于 5.1%。所有契约内外点都保留。",
              "", "合规/成本图：柱为应用×场景合规宏平均，成本 ECDF 每 trace 总权重 1/3；未匹配/无效项位于 +∞ 并注明质量。E5 图只比较相同已锁索引在五个预测器种子下的变化。图为内部科研评估用途，未声称符合特定期刊最终版式。",
              "", "## 可复现产物", ""]
    lines += [f"- [{path.name}]({path.as_posix()})" for path in output_paths]
    lines += ["", "技能贡献：Ponytail 复用纯 validation 选择函数、SQLite 和标准库；scientific-visualization 明确每点来源、非支配定义、坐标及缺失质量；nature-statistics 保留应用/trace层级、重复测量与区间解释；CCFA Integrity 由同一选择和聚合函数派生表格、图和 A/B 计算。"]
    if incomplete:
        lines += ["", "## 未齐记录", ""] + ["- " + str(reason) for reason in incomplete[:100]]
        if len(incomplete) > 100:
            lines.append(f"- 其余 {len(incomplete) - 100} 项同类缺失未逐条展开；原始数据库与候选表可定位。")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "work/gate2.sqlite")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/gate2.toml")
    parser.add_argument("--self-check", action="store_true", help="Synthetic, validation-only checks; never opens a result DB")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    config_path = args.config.resolve()
    require_frozen(ROOT, config_path)
    config = tomllib.loads(config_path.read_text(encoding="utf-8-sig"))
    results_dir, figures_dir = ROOT / "results", ROOT / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    connection = sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    apps = [tuple(row) for row in connection.execute("SELECT DISTINCT trace,app FROM runs ORDER BY trace,app")]
    incomplete = []
    for trace in config["design"]["traces"]:
        actual = sum(t == trace for t, _ in apps)
        if actual != config["design"]["apps_per_trace"]:
            incomplete.append(f"{trace}: {actual}/{config['design']['apps_per_trace']} applications present")
    unexpected = {t for t, _ in apps} - set(config["design"]["traces"])
    if unexpected:
        raise ValueError("Unexpected source traces: " + str(unexpected))
    configurations = [dict(row) for row in connection.execute("SELECT * FROM configs ORDER BY method,candidate")]
    config_counts = Counter(r["method"] for r in configurations)
    for method in METHODS:
        if config_counts[method] != 30:
            incomplete.append(f"{method}: configs table has {config_counts[method]}/30 slots")
    write_csv(results_dir / "gate2-configurations.csv", configurations)

    fields = ("trace", "app", "predictor", "seed", "phase", "budget_mode", "scenario", "method", "candidate",
              "internal_risk", "target", "status", "repetitions", "expected_repetitions", "risk", "cost", "paid",
              "ready", "actions", "minutes", "contract", "physical_floor", "cdt_bound", "mae",
              "within_method_frontier", "all_method_frontier")
    units, app_data, locked, points, hindsight = [], [], [], [], []
    slots = defaultdict(list)
    main_seed = config["design"]["seeds"][0]

    def load(trace, app, predictor, seed):
        partitions = defaultdict(list)
        for row in connection.execute("SELECT * FROM runs WHERE trace=? AND app=? AND predictor=? AND seed=?",
                                      (trace, app, predictor, seed)):
            partitions[row["phase"], row["budget_mode"]].append(dict(row))
        return partitions

    with gzip.open(results_dir / "gate2-candidates.csv.gz", "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()

        def record_candidates(identity, phase, mode, aggregates, plan, base_tests=None):
            for scene in SCENARIOS:
                items = [r for (s, _, _), r in aggregates.items() if s == scene]
                all_front = frontier_keys(items)
                method_front = {m: frontier_keys(r for r in items if r["method"] == m) for m in METHODS}
                denominator = None
                if base_tests is not None:
                    denominator = _stat(base_tests, scene, "HPA", plan[scene]["HPA"]["deployment"])
                for item in items:
                    key = item["method"], item["candidate"]
                    record = dict(identity, phase=phase, budget_mode=mode, **item,
                                  within_method_frontier=int(key in method_front[item["method"]]),
                                  all_method_frontier=int(key in all_front))
                    writer.writerow({k: csv_value(record.get(k)) for k in fields})
                    if (phase, mode, identity["predictor"], identity["seed"]) == ("test", "causal", "lightgbm", main_seed):
                        if item["status"] == "OK" and denominator and denominator["status"] == "OK" and denominator["cost"] > 0:
                            points.append((scene, item["method"], item["cost"] / denominator["cost"], item["risk"],
                                           key in all_front, key in method_front[item["method"]], item["contract"]))
                    if phase == "test" and mode == "hindsight":
                        hindsight.append(dict(record,
                            locked_deployment=int(item["candidate"] == plan[scene][item["method"]]["deployment"]),
                            excess_risk=item["risk"] - item["physical_floor"] if item["status"] == "OK" else math.nan))

        def check_slots(identity, phase, aggregates, required):
            missing = required - set(aggregates)
            if missing:
                incomplete.append(f"{identity}/{phase}: {len(missing)} required candidate/case slots missing")
            bad = [key for key in required & set(aggregates)
                   if aggregates[key]["status"] not in ("OK", "NO_RANK", "NONDISCRIMINATING")]
            if bad:
                incomplete.append(f"{identity}/{phase}: {len(bad)} incomplete/unexpected candidate statuses; first={bad[0]}")

        def add_unit(identity, validation, tests, plan, reference=None):
            unit = summarize_unit(**identity, validation=validation, tests=tests, plan=plan,
                                  config=config, contract_reference=reference)
            units.append(unit)
            app_data.extend(application_rows(unit, config))
            for scene in SCENARIOS:
                for method in METHODS:
                    for mode, candidate in plan[scene][method].items():
                        val = _stat(validation, scene, method, candidate)
                        test = _stat(tests, scene, method, candidate)
                        locked.append(dict(identity, scenario=scene, method=method, mode=mode, candidate=candidate,
                                           status=test["status"] if test else "MISSING",
                                           validation_risk=val["risk"] if val else math.nan,
                                           validation_cost=val["cost"] if val else math.nan,
                                           test_risk=test["risk"] if test else math.nan,
                                           test_cost=test["cost"] if test else math.nan,
                                           contract=int(unit["cases"][scene]["contract"]),
                                           physical_floor=unit["cases"][scene]["physical_floor"],
                                           ratio=unit["cases"][scene]["ratios"].get(method) if mode == "matched" else None,
                                           match_status=unit["cases"][scene]["match_reason"].get(method) if mode == "matched" else None))
            return unit

        full_grid = {(s, m, c) for s in SCENARIOS for m in METHODS for c in range(30)}
        for position, (trace, app) in enumerate(apps):
            reference_unit = None
            for predictor, seed in (("lightgbm", main_seed), ("persistence", 0)):
                identity = dict(trace=trace, app=app, predictor=predictor, seed=seed)
                partitions = load(**identity)
                val_rows = partitions.get(("validation", "causal"), [])
                validation = aggregate_rows(val_rows, config)
                tests = aggregate_rows(partitions.get(("test", "causal"), []), config)
                plan = selection_plan(val_rows, config)
                check_slots(identity, "validation", validation, full_grid)
                check_slots(identity, "test", tests, full_grid)
                for method in METHODS:
                    slots[method].append(sum(r["status"] == "OK" for (s, m, _), r in validation.items()
                                             if s == "nominal" and m == method))
                record_candidates(identity, "validation", "causal", validation, plan)
                record_candidates(identity, "test", "causal", tests, plan, tests)
                unit = add_unit(identity, validation, tests, plan, reference_unit)
                hindsight_aggregates = {}
                for (phase, mode), values in partitions.items():
                    if mode == "causal":
                        continue
                    aggregates = aggregate_rows(values, config)
                    record_candidates(identity, phase, mode, aggregates, plan)
                    if (phase, mode) == ("test", "hindsight"):
                        hindsight_aggregates.update(aggregates)
                expected_hindsight = {(scene, method, c) for scene in SCENARIOS
                                      if finite(unit["cases"][scene]["physical_floor"])
                                      and unit["cases"][scene]["physical_floor"] > 0
                                      for method in ("CDT", "BACC-lite", "PAC-h") for c in range(30)}
                check_slots(identity, "hindsight test", hindsight_aggregates, expected_hindsight)
                # Non-risk controllers reuse the causal replay; mark its hindsight information explicitly.
                for (scene, method, candidate), item in tests.items():
                    floor = unit["cases"][scene]["physical_floor"]
                    if method in ("HPA", "Point+margin") and finite(floor) and floor > 0:
                        target = item["_diagnostic_target"]
                        hindsight.append({**identity, **item, "phase": "test", "budget_mode": "hindsight_causal_reuse",
                            "target": target, "status": "NONDISCRIMINATING"
                            if item["_diagnostic_nondiscriminating"] else item["status"],
                            "locked_deployment": int(candidate == plan[scene][method]["deployment"]),
                            "excess_risk": item["risk"] - floor if item["status"] == "OK" else math.nan})
                if predictor == "lightgbm":
                    reference_unit = unit
                    selected = selection_sets(val_rows, config)
                    required = {(scene, method, candidate) for scene, methods in selected.items()
                                for method, candidates in methods.items() for candidate in candidates}
                    for extra_seed in config["design"]["seeds"][1:]:
                        extra_identity = dict(trace=trace, app=app, predictor=predictor, seed=extra_seed)
                        extra = load(**extra_identity)
                        extra_test = aggregate_rows(extra.get(("test", "causal"), []), config)
                        check_slots(extra_identity, "test", extra_test, required)
                        record_candidates(extra_identity, "test", "causal", extra_test, plan)
                        add_unit(extra_identity, validation, extra_test, plan, reference_unit)
            if (position + 1) % 10 == 0:
                print(f"Analyzed {position + 1}/{len(apps)} applications", flush=True)
    connection.close()
    write_csv(results_dir / "gate2-locked.csv.gz", locked)
    write_csv(results_dir / "gate2-applications.csv", app_data)
    write_csv(results_dir / "gate2-hindsight.csv.gz", hindsight, list(fields) + ["locked_deployment", "excess_risk"])
    print("Computing application/trace summaries and bootstrap intervals", flush=True)
    summary, decisions = build_summary(app_data, units, config)
    paired = paired_statistics(app_data, units, config)
    write_csv(results_dir / "gate2-summary.csv", summary)
    write_csv(results_dir / "gate2-paired-statistics.csv", paired)
    print("Rendering three figure families", flush=True)
    make_figures(points, app_data, summary, config, figures_dir)
    paths = [Path("results") / name for name in (
        "gate2-configurations.csv", "gate2-candidates.csv.gz", "gate2-locked.csv.gz", "gate2-applications.csv",
        "gate2-hindsight.csv.gz", "gate2-summary.csv", "gate2-paired-statistics.csv")]
    paths += [Path("figures") / f"{stem}.{suffix}" for stem in (
        "gate2-frontiers", "gate2-compliance-cost", "gate2-seed-sensitivity") for suffix in ("png", "pdf")]
    report = report_text(config, summary, decisions, units, slots, incomplete, hindsight, paths)
    import matplotlib
    import scipy
    report += f"\n分析软件：Python {sys.version.split()[0]}；SciPy {scipy.__version__}；Matplotlib {matplotlib.__version__}。\n"
    (ROOT / "docs/gate2-results.md").write_text(report, encoding="utf-8")
    print("Analysis written to docs/gate2-results.md; final Gate remains the user's decision.", flush=True)


if __name__ == "__main__":
    main()



