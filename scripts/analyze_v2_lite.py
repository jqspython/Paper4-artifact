"""Prospective v2-lite estimands. No trace IO; real CSV analysis requires freeze gates."""

import argparse
import csv
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_gate2 import weighted_median
from v2_lite import BASES, PRIMARY, SOURCES, load_config, require_stage_gate

VENDORS = {"Huawei": SOURCES[:2], "Azure": SOURCES[2:]}
VARIANTS = ("primary", "ITT", "tolerance_0.001", "tolerance_0.005",
            "tolerance_0.02", "mixed_median", "Huawei_source_slots",
            "total_cost", "validation_budget_met")


class Incomplete(ValueError):
    """A required selected application, pair or observation is unavailable."""


def numeric(value):
    return float(value) if value not in (None, "", "NA") else math.nan


def panels(rows, expected, eta, stage):
    """Source arrays: app x case x base x {base,wrapper} x {risk,cost}.

    The independently supplied ID manifest, not successful rows, fixes denominators.
    Validation rows and descriptive cases never enter these endpoint arrays.
    """
    if set(expected) != set(SOURCES) or any(not ids or len(set(ids)) != len(ids)
                                           for ids in expected.values()):
        raise Incomplete("Manifest must contain distinct IDs in all three sources")
    result = {s: dict(ids=tuple(map(str, expected[s])),
                      values=np.full((len(expected[s]), len(PRIMARY), len(BASES), 2, 2), np.nan),
                      paid=np.full((len(expected[s]), len(PRIMARY), len(BASES), 2), np.nan),
                      available=np.full((len(expected[s]), len(BASES)), -1, dtype=int),
                      validation_ok=np.full((len(expected[s]), len(BASES)), -1, dtype=int),
                      contract=np.full((len(expected[s]), len(PRIMARY)), -1, dtype=int)) for s in SOURCES}
    lookup = {s: {str(app): i for i, app in enumerate(expected[s])} for s in SOURCES}
    seen, locked, durations = set(), {}, {}
    for row in rows:
        if row.get("stage") != stage or row.get("phase") != "test" or row.get("role") != "locked":
            continue
        if row["base"] not in BASES or row["scenario"] not in PRIMARY:
            continue
        wrapped = int(row["wrapped"])
        if wrapped not in (0, 1):
            raise Incomplete("Invalid wrapper flag")
        if wrapped and not math.isclose(numeric(row["eta"]), eta, abs_tol=1e-12):
            continue
        source, app = row["source"], str(row["app"])
        if source not in lookup or app not in lookup[source]:
            raise Incomplete("Unexpected source-qualified application in result CSV")
        a, c, b = lookup[source][app], PRIMARY.index(row["scenario"]), BASES.index(row["base"])
        key = (source, a, c, b, wrapped)
        if key in seen or int(row["replicate"]) != 0:
            raise Incomplete("Duplicate or unexpected primary episode")
        seen.add(key)
        flag = int(row["contract"])
        if flag not in (0, 1) or result[source]["contract"][a, c] not in (-1, flag):
            raise Incomplete("Contract mask must be method-independent")
        result[source]["contract"][a, c] = flag
        validation_status = row.get("validation_status")
        unavailable = row["status"] == validation_status == "NO_VALID_CONFIG"
        if not unavailable and (row["status"] != "OK" or validation_status not in ("OK", "VALIDATION_BUDGET_NOT_MET")):
            raise Incomplete(f"Necessary legal run failed or validation state missing: {source}/{app}/{row['base']}: {row['status']}")
        state, validation_ok = int(not unavailable), int(validation_status == "OK")
        panel = result[source]
        if panel["available"][a, b] not in (-1, state) or panel["validation_ok"][a, b] not in (-1, validation_ok):
            raise Incomplete("Availability and validation status must describe the entire app/base pair")
        panel["available"][a, b], panel["validation_ok"][a, b] = state, validation_ok
        duration = numeric(row["minutes"])
        if not math.isfinite(duration) or duration <= 0:
            raise Incomplete("Invalid observed duration")
        exposure = (source, a, c)
        if exposure in durations and duration != durations[exposure]:
            raise Incomplete("Paired methods must cover the same episode duration")
        durations[exposure] = duration
        if unavailable:
            continue
        loss, paid = map(numeric, (row["violations"], row["paid"]))
        if not (math.isfinite(loss) and 0 <= loss <= duration and math.isfinite(paid) and paid > 0):
            raise Incomplete("Invalid observed violations or paid replicas")
        panel["values"][a, c, b, wrapped] = (loss / duration, paid / duration)
        panel["paid"][a, c, b, wrapped] = paid
        signature = tuple(str(row.get(k, "")) for k in
                          ("candidate", "config_risk", "parameter", "window", "stable"))
        base_key = (source, a, b)
        if base_key in locked and locked[base_key] != signature:
            raise Incomplete("Base and wrapper must share one nominal lock across cases")
        locked[base_key] = signature
    required = sum(len(ids) for ids in expected.values()) * len(PRIMARY) * len(BASES) * 2
    if len(seen) != required or any((p["contract"] < 0).any() or (p["available"] < 0).any()
                                   or (p["validation_ok"] < 0).any() for p in result.values()):
        raise Incomplete("Missing selected application, method, case or paired episode")
    for p in result.values():
        p["available"], p["validation_ok"] = p["available"].astype(bool), p["validation_ok"].astype(bool)
    return result


def source_metrics(panel, tolerance, budget, itt=False, validation_budget_met=False):
    values = panel["values"]
    mask = np.ones_like(panel["contract"], dtype=bool) if itt else panel["contract"].astype(bool)
    if not mask.any(axis=0).all():
        raise Incomplete("Empty source/case denominator")
    available = np.asarray(panel.get("available", np.ones((len(mask), len(BASES)), dtype=bool)))
    validation_ok = np.asarray(panel.get("validation_ok", np.ones_like(available)))
    paid = panel.get("paid", values[..., 1])  # Unit-duration synthetic panels remain a small test API.
    compliant = values[..., 0] <= budget + tolerance + 1e-12
    compliance, ratios = np.full((len(BASES), 2), np.nan), np.full(len(BASES), np.nan)
    app_ratios = np.full((len(mask), len(BASES)), np.nan)
    app_deltas, app_totals = app_ratios.copy(), app_ratios.copy()
    totals = np.zeros((len(BASES), 2))
    effective = np.zeros((len(PRIMARY), len(BASES)), dtype=int)
    for b in range(len(BASES)):
        use = available[:, b] & (validation_ok[:, b] if validation_budget_met else True)
        valid = mask & use[:, None]
        effective[:, b] = valid.sum(axis=0)
        if effective[:, b].all():
            compliance[b] = (np.where(valid[..., None], compliant[:, :, b], 0).sum(axis=0)
                             / effective[:, b, None]).mean(axis=0)
        included = valid.any(axis=1)
        if not included.any():
            continue
        cost = values[:, :, b, :, 1]
        case_ratios = np.divide(cost[..., 1], cost[..., 0], out=np.full(mask.shape, np.nan), where=valid)
        app_ratios[included, b] = np.nanmedian(case_ratios[included], axis=1)
        ratios[b] = np.median(app_ratios[included, b])
        difference = compliant[:, :, b, 1].astype(int) - compliant[:, :, b, 0].astype(int)
        app_deltas[included, b] = np.where(valid, difference, 0).sum(axis=1)[included] / valid.sum(axis=1)[included]
        app_paid = np.where(valid[..., None], paid[:, :, b], 0).sum(axis=1)
        totals[b] = app_paid.sum(axis=0)
        app_totals[included, b] = app_paid[included, 1] / app_paid[included, 0]
    return dict(compliance=compliance, ratios=ratios, app_ratios=app_ratios, app_deltas=app_deltas,
                app_totals=app_totals, total_paid=totals, available=available,
                unavailable=(~available).sum(axis=0), unavailable_fraction=(~available).mean(axis=0),
                validation_excluded=(available & ~validation_ok).sum(axis=0), effective_support=effective,
                empty_base=(effective == 0).any(axis=0),
                without_contract=int((~mask.any(axis=1)).sum()), support=mask.sum(axis=0))


def estimate(data, config, variant="primary", source_slots=None):
    if variant not in VARIANTS:
        raise ValueError("Unknown sensitivity variant")
    tolerance = float(variant.split("_", 1)[1]) if variant.startswith("tolerance_") else config["endpoints"]["compliance_tolerance"]
    by_source = {s: source_metrics(data[s], tolerance, config["gate"]["budget"], variant == "ITT",
                                   variant == "validation_budget_met")
                 for s in SOURCES}
    by_vendor = {}
    for vendor, sources in VENDORS.items():
        slots = tuple(source_slots) if vendor == "Huawei" and source_slots is not None else sources
        compliance = np.mean([by_source[s]["compliance"] for s in slots], axis=0)
        ratios = np.mean([by_source[s]["ratios"] for s in slots], axis=0)
        if variant == "mixed_median":
            for b in range(len(BASES)):
                groups = [m["app_ratios"][:, b][np.isfinite(m["app_ratios"][:, b])]
                          for m in (by_source[s] for s in slots)]
                ratios[b] = weighted_median(
                    [v for group in groups for v in group],
                    [1 / len(groups) / len(group) for group in groups for _ in group]) if all(len(g) for g in groups) else math.nan
        # F2 is an actual-spend sensitivity: sums, not source-weighted means.
        totals = np.sum([by_source[s]["total_paid"] for s in slots], axis=0)
        if variant == "total_cost":
            ratios = np.divide(totals[:, 1], totals[:, 0], out=np.full(len(BASES), np.nan), where=totals[:, 0] > 0)
        by_vendor[vendor] = dict(compliance=compliance, ratios=ratios, total_paid=totals)
    average = {key: np.mean([v[key] for v in by_vendor.values()], axis=0)
               for key in ("compliance", "ratios")}
    def flatten(item):
        comp, ratios = item["compliance"], item["ratios"]
        delta = comp[:, 1] - comp[:, 0]
        def cost(indices):
            if variant != "total_cost":
                return float(ratios[indices].mean())
            totals = item["total_paid"][indices].sum(axis=0)
            return float(totals[1] / totals[0]) if totals[0] > 0 else math.nan
        return dict(p1_delta=float(delta[:2].mean()), p1_cost_ratio=cost(slice(0, 2)),
                    p2_cost_ratio=cost(slice(2, 4)), p2_delta=float(delta[2:].mean()),
                    p3_compliance=float(comp[:, 1].mean()), all_cost_ratio=cost(slice(0, 4)),
                    **{f"wrapper_{base}": float(comp[b, 1]) for b, base in enumerate(BASES)})
    vendor_points = {v: flatten(item) for v, item in by_vendor.items()}
    point = {key: float(np.mean([v[key] for v in vendor_points.values()])) for key in next(iter(vendor_points.values()))}
    if variant == "total_cost":
        for vendor, sources in VENDORS.items():
            for label, indices in (("p1", slice(0, 2)), ("p2", slice(2, 4))):
                totals = np.concatenate([by_source[s]["app_totals"][:, indices].ravel() for s in sources])
                totals = totals[np.isfinite(totals)]
                vendor_points[vendor][label + "_app_total_ratio_p90"] = float(np.quantile(
                    totals, config["statistics"].get("cost_tail_quantile", .9), method="linear")) if len(totals) else math.nan
    threshold = config["statistics"].get("maximum_unavailable_fraction", .05)
    invalid = np.any([m["unavailable_fraction"] > threshold + 1e-12 for m in by_source.values()], axis=0)
    empty = np.any([m["empty_base"] for m in by_source.values()], axis=0)
    assessable = {"P1": not (invalid[:2] | empty[:2]).any(),
                  "P2": not (invalid[2:] | empty[2:]).any(), "P3": not (invalid | empty).any()}
    return dict(point=point, vendors=vendor_points, sources=by_source, aggregate=average,
                vendor_arrays=by_vendor, assessable=assessable)


def bootstrap(data, config, variant="primary"):
    stats = config["statistics"]
    slots = variant == "Huawei_source_slots"
    rng = np.random.default_rng(stats["source_slot_sensitivity_seed"] if slots else stats["bootstrap_seed"])
    original = estimate(data, config, variant)["point"]
    draws = {key: [] for key, value in original.items() if math.isfinite(value)}
    attempts = 0
    def resample(panel):
        ix = rng.integers(0, len(panel["ids"]), len(panel["ids"]))
        return dict(ids=tuple(panel["ids"][i] for i in ix),
                    **{key: panel[key][ix] for key in ("values", "contract", "paid", "available", "validation_ok") if key in panel})
    while draws and any(len(values) < stats["bootstrap_repetitions"] for values in draws.values()) and attempts < stats["maximum_bootstrap_attempts"]:
        attempts += 1
        sampled = {}
        # One draw per source carries every case/method/pair; sources remain fixed.
        for source, panel in data.items():
            sampled[source] = resample(panel)
        try:
            if slots:
                # Repeated source slots get independent application resamples.
                selected = tuple(rng.choice(SOURCES[:2], size=2))
                for j, source in enumerate(selected):
                    sampled[f"slot{j}"] = resample(data[source])
                # estimate accepts named slots: alias temporary source panels without changing vendor weights.
                slot_data = {SOURCES[0]: sampled["slot0"], SOURCES[1]: sampled["slot1"], SOURCES[2]: sampled[SOURCES[2]]}
                point = estimate(slot_data, config, "primary")["point"]
            else:
                point = estimate(sampled, config, variant)["point"]
            # Preserve local estimability: an unavailable O metric cannot discard a U draw.
            for key, values in draws.items():
                if len(values) < stats["bootstrap_repetitions"] and math.isfinite(point[key]):
                    values.append(point[key])
        except Incomplete:
            continue
    counts = {key: len(draws.get(key, ())) for key in original}
    intervals = {key: tuple(map(float, np.quantile(values, [.025, .975], method="linear")))
                 for key, values in draws.items() if len(values) == stats["bootstrap_repetitions"]}
    statuses = {key: "OK" if key in intervals else "CI_UNAVAILABLE" for key in original}
    status = "OK" if all(value == "OK" for value in statuses.values()) else "PARTIAL" if intervals else "CI_UNAVAILABLE"
    return dict(status=status, attempts=attempts, valid=min((len(v) for v in draws.values()), default=0),
                intervals=intervals, metric_status=statuses, metric_valid=counts)


def verdict(result, interval, config):
    e, p, vendors = config["endpoints"], result["point"], result["vendors"]
    p1_ci = interval.get("metric_status", {}).get("p1_delta", interval.get("status")) == "OK"
    improvement = (p["p1_delta"] >= e["p1_improvement"] - 1e-12
                   and p1_ci
                   and interval["intervals"]["p1_delta"][0] > 0
                   and all(v["p1_delta"] > 0 for v in vendors.values()))
    guard = p["p1_cost_ratio"] <= e["p1_cost_guardrail"] + 1e-12
    p1 = improvement and guard
    p2 = (p["p2_cost_ratio"] <= e["p2_median_cost_ratio"] + 1e-12
          and p["p2_delta"] >= e["p2_minimum_compliance_change"] - 1e-12
          and all(v["p2_cost_ratio"] < 1 for v in vendors.values()))
    p3 = (p["p3_compliance"] >= e["p3_macro_compliance"] - 1e-12
          and all(p[f"wrapper_{base}"] >= e["p3_individual_wrapper_compliance"] - 1e-12 for base in BASES))
    eligible = result.get("assessable", dict.fromkeys(("P1", "P2", "P3"), True))
    out = dict(P1="INCOMPLETE" if not eligible["P1"] or not p1_ci else
               "PASS" if p1 else "ONLY_BY_OVERPROVISIONING" if improvement and not guard else "FAIL",
               P2="INCOMPLETE" if not eligible["P2"] else "PASS" if p2 else "FAIL",
               P3="INCOMPLETE" if not eligible["P3"] else "PASS" if p3 else "FAIL")
    out["decision"] = ("INCOMPLETE" if out["P3"] == "INCOMPLETE" else "STOP_WRAPPER_SAFETY" if not p3 else
                       "INCOMPLETE" if "INCOMPLETE" in out.values() else
                       "BOTH_CATEGORIES" if p1 and p2 else "PROTECTION_ONLY" if p1 else
                       "BUDGET_RECOVERY_ONLY" if p2 else "STOP_WRAPPER_VALUE")
    return out


def select_eta(by_eta, config):
    candidates = []
    for eta in config["search"]["wrapper_eta"]:
        result = estimate(by_eta[eta], config)
        p = result["point"]
        support_ok = all(m["support"].min() >= config["gate"]["minimum_contract_apps_per_trace"] for m in result["sources"].values())
        if (support_ok and result["assessable"]["P3"] and math.isfinite(p["all_cost_ratio"])
                and p["p3_compliance"] >= config["endpoints"]["p3_macro_compliance"] - 1e-12
                and all(p[f"wrapper_{b}"] >= config["endpoints"]["p3_individual_wrapper_compliance"] - 1e-12 for b in BASES)):
            candidates.append((p["all_cost_ratio"], eta))
    if not candidates:
        return None
    best = min(cost for cost, _ in candidates)
    return min(eta for cost, eta in candidates if cost <= best + 1e-10)


def paired_tests(data, config):
    tests = []
    for source in SOURCES:
        m = source_metrics(data[source], config["endpoints"]["compliance_tolerance"], config["gate"]["budget"])
        for b, base in enumerate(BASES):
            differences = m["app_deltas"][:, b] if b < 2 else np.log(m["app_ratios"][:, b])
            differences = differences[np.isfinite(differences)]
            nonzero = differences[differences != 0]
            p = (math.nan if not len(differences) else
                 float(wilcoxon(nonzero, alternative="two-sided", method="approx").pvalue) if len(nonzero) else 1.)
            tests.append(dict(source=source, base=base, n=len(differences), p=p))
    running = 0.
    for test in tests:
        test["holm_p"] = math.nan
    for i, test in enumerate(sorted((t for t in tests if math.isfinite(t["p"])), key=lambda t: t["p"])):
        running = max(running, min(1., (len(tests) - i) * test["p"]))
        test["holm_p"] = running
    return tests


def source_rows(result, variant, config):
    rows = []
    for source, m in result["sources"].items():
        for b, base in enumerate(BASES):
            scope = f"{source}/{base}"
            ratio = m["ratios"][b]
            if variant == "total_cost":
                ratio = m["total_paid"][b, 1] / m["total_paid"][b, 0] if m["total_paid"][b, 0] > 0 else math.nan
            metrics = dict(base_compliance=m["compliance"][b, 0], wrapped_compliance=m["compliance"][b, 1],
                           cost_ratio=ratio, unavailable_apps=int(m["unavailable"][b]),
                           unavailable_fraction=float(m["unavailable_fraction"][b]),
                           available_apps=int(m["available"][:, b].sum()),
                           validation_budget_not_met_apps=int(m["validation_excluded"][b]))
            if variant == "total_cost":
                totals = m["app_totals"][:, b]
                totals = totals[np.isfinite(totals)]
                metrics["app_total_cost_ratio_p90"] = float(np.quantile(totals,
                    config["statistics"].get("cost_tail_quantile", .9), method="linear")) if len(totals) else math.nan
                metrics["paired_paid_base"], metrics["paired_paid_wrapped"] = m["total_paid"][b]
            rows += [dict(variant=variant, scope=scope, metric=k, value=float(v)) for k, v in metrics.items()]
            rows += [dict(variant=variant, scope=f"{scope}/{case}", metric="effective_paired_apps",
                          value=int(m["effective_support"][c, b])) for c, case in enumerate(PRIMARY)]
    return rows


def report(data, config):
    support = {s: data[s]["contract"].sum(axis=0).tolist() for s in SOURCES}
    if any(min(counts) < config["gate"]["minimum_contract_apps_per_trace"] for counts in support.values()):
        rows = status_rows("NOT_ASSESSABLE")
        availability = estimate(data, config, "ITT")
        rows += [row for row in source_rows(availability, "availability", config)
                 if row["metric"] in ("unavailable_apps", "unavailable_fraction", "available_apps",
                                       "validation_budget_not_met_apps")]
        for source, counts in support.items():
            for j, case in enumerate(PRIMARY):
                rows.append(dict(variant="support", scope=f"{source}/{case}", metric="contract_apps", value=counts[j]))
                rows.append(dict(variant="support", scope=f"{source}/{case}", metric="excluded_fraction",
                                 value=float(1 - data[source]["contract"][:, j].mean())))
        return rows, None
    rows, main_verdict = [], None
    for variant in VARIANTS:
        point = estimate(data, config, variant)
        ci = bootstrap(data, config, variant)
        if variant == "primary":
            main_verdict = verdict(point, ci, config)
        for scope, values in (("overall", point["point"]), *point["vendors"].items()):
            for metric, value in values.items():
                low, high = ci["intervals"].get(metric, (math.nan, math.nan)) if scope == "overall" else (math.nan, math.nan)
                rows.append(dict(variant=variant, scope=scope, metric=metric, value=value,
                                 ci_low=low, ci_high=high,
                                 ci_status=ci.get("metric_status", {}).get(metric, "CI_UNAVAILABLE") if scope == "overall" else "CI_UNAVAILABLE",
                                 bootstrap_valid=ci.get("metric_valid", {}).get(metric, 0) if scope == "overall" else 0,
                                 bootstrap_attempts=ci["attempts"] if scope == "overall" else 0))
        rows += source_rows(point, variant, config)
        rows += [dict(variant=variant, scope="overall", metric=endpoint + "_availability",
                      value="OK" if valid else "INCOMPLETE") for endpoint, valid in point["assessable"].items()]
    for source, panel in data.items():
        for j, case in enumerate(PRIMARY):
            rows.append(dict(variant="support", scope=f"{source}/{case}", metric="contract_apps", value=support[source][j]))
            rows.append(dict(variant="support", scope=f"{source}/{case}", metric="excluded_fraction", value=float(1 - panel["contract"][:, j].mean())))
        rows.append(dict(variant="support", scope=source, metric="no_contract_primary_case", value=int((~panel["contract"].any(axis=1)).sum())))
    for test in paired_tests(data, config):
        for metric in ("p", "holm_p", "n"):
            rows.append(dict(variant="secondary_Wilcoxon", scope=f"{test['source']}/{test['base']}", metric=metric, value=test[metric]))
    for metric, value in main_verdict.items():
        rows.append(dict(variant="primary", scope="overall", metric=metric, value=value))
    return rows, main_verdict


def status_rows(status):
    return [dict(variant=variant, scope="overall", metric="status", value=status) for variant in VARIANTS]


def exploration_rows(by_eta, config, selected):
    """All five operating points; stage-labelled diagnostics, never confirmatory claims."""
    rows = []
    for eta, data in sorted(by_eta.items()):
        result = estimate(data, config)
        point, limit = result["point"], config["endpoints"]
        support_ok = all(m["support"].min() >= config["gate"]["minimum_contract_apps_per_trace"]
                         for m in result["sources"].values())
        eligible = (support_ok and result["assessable"]["P3"] and math.isfinite(point["all_cost_ratio"])
                    and point["p3_compliance"] >= limit["p3_macro_compliance"] - 1e-12
                    and all(point[f"wrapper_{b}"] >= limit["p3_individual_wrapper_compliance"] - 1e-12 for b in BASES))
        reason = ("INSUFFICIENT_CONTRACT_SUPPORT" if not support_ok else
                  "INCOMPLETE_AVAILABILITY" if not result["assessable"]["P3"] else
                  "SAFETY_CRITERION_NOT_MET" if not eligible else
                  "SELECTED_MINIMUM_COST_WITH_SMALLER_ETA_TIEBREAK" if eta == selected else "ELIGIBLE_NOT_SELECTED")
        variant = f"eta_{eta}"
        for scope, values in (("overall", point), *result["vendors"].items()):
            rows += [dict(variant=variant, scope=scope, metric=k, value=v) for k, v in values.items()]
        for scope, values in (("overall", result["aggregate"]), *result["vendor_arrays"].items()):
            for b, base in enumerate(BASES):
                for metric, value in (("base_compliance", values["compliance"][b, 0]),
                                      ("wrapped_compliance", values["compliance"][b, 1]),
                                      ("cost_ratio", values["ratios"][b])):
                    rows.append(dict(variant=variant, scope=f"{scope}/{base}", metric=metric, value=float(value)))
        rows += source_rows(result, variant, config)
        for metric, value in (("selection_eligible", int(eligible)), ("selection_reason", reason),
                              ("selected", int(eta == selected)),
                              ("grid_boundary_limitation", int(eta == selected and eta in (
                                  min(config["search"]["wrapper_eta"]), max(config["search"]["wrapper_eta"]))))):
            rows.append(dict(variant=variant, scope="overall", metric=metric, value=value))
    return rows


def all_case_support(rows, expected, stage, config):
    """Descriptive support uses no method's success to define an exclusion mask."""
    flags = {}
    selected = {s: set(map(str, ids)) for s, ids in expected.items()}
    for row in rows:
        if row.get("stage") != stage or row.get("phase") != "test" or row.get("role") != "locked":
            continue
        source, app, case = row["source"], str(row["app"]), row["scenario"]
        if source not in expected or app not in selected[source]:
            raise Incomplete("Unexpected application in descriptive support")
        flag = int(row["contract"])
        key = source, app, case, int(row["replicate"])
        if key in flags and flags[key] != flag:
            raise Incomplete("Descriptive contract flag depends on method or eta")
        if flag not in (0, 1):
            raise Incomplete("Unknown descriptive contract flag")
        flags[key] = flag
    output = []
    # All 13 cases are fixed, including the two five-position flash cases.
    cases = ("nominal", *PRIMARY, "ramp1.5", "ramp2", "step1.5", "step2",
             "flash3x15", "flash5x60", "rate1", "cap8_rate1")
    for source in SOURCES:
        for case in cases:
            reps = config["design"]["seeds"] if case.startswith("flash") else (0,)
            covered = [all(flags[(source, str(app), case, rep)] for rep in reps)
                       for app in expected[source]
                       if all((source, str(app), case, rep) in flags for rep in reps)]
            complete = len(covered) == len(expected[source])
            planned = not (stage == "explore" and config.get("execution", {}).get("exploration_scope") == "primary"
                           and case not in PRIMARY)
            for metric, value in (("status", "OK" if complete else "NOT_RUN_EXPLORATION_SCOPE" if not planned else "INCOMPLETE"),
                                  ("selected_apps", len(expected[source])), ("observed_apps", len(covered)),
                                  ("contract_apps", sum(covered)),
                                  ("excluded_fraction", 1 - sum(covered) / len(covered) if complete else "NA")):
                output.append(dict(variant="all_case_support", scope=f"{source}/{case}", metric=metric, value=value))
    return output


def endpoint_destination(stage):
    """Confirmation never overwrites the already archived exploratory endpoint CSV."""
    return ROOT / "results" / ("v2-lite-confirm-endpoints.csv" if stage == "confirm" else "v2-lite-endpoints.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("explore", "confirm"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v2-lite.toml")
    args = parser.parse_args()
    config = load_config(args.config)
    reference = require_stage_gate(config, args.stage)  # Before results or IDs are read.
    from v2_lite import validate_manifest
    expected = validate_manifest(config, args.stage)
    with (ROOT / config["io"]["summary_csv"]).open(encoding="utf-8-sig", newline="") as handle:
        runs = list(csv.DictReader(handle))
    if any(r["config_ref"] != reference for r in runs if r["stage"] == args.stage):
        raise Incomplete("Results do not refer to the required frozen source")
    try:
        etas = config["search"]["wrapper_eta"] if args.stage == "explore" else [config["frozen_numeric"]["selected_eta"]]
        datasets = {eta: panels(runs, expected, eta, args.stage) for eta in etas}
        eta = select_eta(datasets, config) if args.stage == "explore" else etas[0]
        if eta is None:
            output, decision = status_rows("EXPLORATORY_SELECTION_FAILED"), None
        else:
            output, decision = report(datasets[eta], config)
        if args.stage == "explore":
            output += exploration_rows(datasets, config, eta)
            output.append(dict(variant="selection", scope="overall", metric="selected_eta", value=eta if eta is not None else "NA"))
    except Incomplete as error:
        output, decision = status_rows(f"INCOMPLETE: {error}"), None
    try:
        output += all_case_support(runs, expected, args.stage, config)
    except Incomplete as error:
        output += [dict(variant="all_case_support", scope="overall", metric="status", value=f"INCOMPLETE: {error}")]
    destination = endpoint_destination(args.stage)
    fields = ("stage", "config_ref", "variant", "scope", "metric", "value", "ci_low", "ci_high", "ci_status", "bootstrap_valid", "bootstrap_attempts")
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows({key: "NA" if isinstance(value, (float, np.floating)) and not math.isfinite(value) else value
                          for key, value in dict(stage=args.stage, config_ref=reference, **row).items()} for row in output)
    print(f"{destination}: {decision or 'NO CONFIRMATORY VERDICT'}")
    if args.stage == "confirm":
        from scripts.report_v2_lite_confirmation import write_confirmation
        write_confirmation(runs, expected, config, reference)


if __name__ == "__main__":
    main()
