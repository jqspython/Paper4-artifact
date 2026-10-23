"""Frozen confirmation descriptions A/B/C. No selection, fitting or control replay.

Risk quantiles use linear interpolation over unweighted app x case cells. Huawei
sources are pooled by actual cells here, unlike the primary source-half estimator.
"""
import csv
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.analyze_gate2 import choose
from scripts.analyze_v2_lite import (BASES, PRIMARY, SOURCES, VENDORS, Incomplete,
                                    exploration_rows, numeric, panels)
from v2_lite import candidate_grid, load_config, require_stage_gate, validate_manifest

CASES = ("nominal", *PRIMARY, "ramp1.5", "ramp2", "step1.5", "step2",
         "flash3x15", "flash5x60", "rate1", "cap8_rate1")
CONTROLLERS = tuple(f"{base}_{kind}" for base in BASES for kind in ("raw", "wrapped")) + ("BACC-lite",)
ORACLE = "known_delay_HPA"


def scopes():
    return [("vendor", name, sources) for name, sources in VENDORS.items()] + [
        ("source", source, (source,)) for source in SOURCES]


def risk_distribution(data, config):
    """A: identical paired, contract-valid support for each raw/wrapped base."""
    output = []
    for level, scope, sources in scopes():
        for b, base in enumerate(BASES):
            masks = {s: data[s]["contract"].astype(bool) & data[s]["available"][:, b, None] for s in sources}
            cells = sum(int(mask.sum()) for mask in masks.values())
            apps = sum(int(mask.any(axis=1).sum()) for mask in masks.values())
            for wrapped in (0, 1):
                values = np.concatenate([data[s]["values"][:, :, b, wrapped, 0][masks[s]] for s in sources])
                quantiles = np.quantile(values, [.25, .5, .75, .9], method="linear") if cells else [math.nan] * 4
                output.append(dict(scope_level=level, scope=scope, base=base,
                    controller=f"{base}_{'wrapped' if wrapped else 'raw'}", eta=.1 if wrapped else None,
                    status="OK" if cells else "EMPTY_PAIRED_SUPPORT", N_app_case_cells=cells,
                    distinct_apps=apps, selected_apps=sum(len(data[s]["ids"]) for s in sources),
                    unavailable_apps=sum(int((~data[s]["available"][:, b]).sum()) for s in sources),
                    q25=quantiles[0], q50=quantiles[1], q75=quantiles[2], q90=quantiles[3],
                    fraction_risk_gt_005=float(np.mean(values > .05)) if cells else math.nan,
                    fraction_risk_gt_006=float(np.mean(values > .06)) if cells else math.nan,
                    pooling="unweighted_app_case_cells_not_independent_n", quantile_method="linear"))
    return output


def controller(row, eta=.1):
    if row["role"] == "oracle_locked":
        return ORACLE if row["base"] == "HPA" and int(row["wrapped"]) == 0 else None
    if row["base"] == "BACC-lite" and int(row["wrapped"]) == 0:
        return "BACC-lite"
    if row["base"] not in BASES:
        return None
    if int(row["wrapped"]) and not math.isclose(numeric(row["eta"]), eta, rel_tol=0, abs_tol=1e-12):
        return None
    return row["base"] + ("_wrapped" if int(row["wrapped"]) else "_raw")


def oracle_audit(rows, expected, config):
    """Recheck that known-delay locks come from all 30 validation candidates."""
    grouped, locked = {}, {}
    for row in rows:
        if row.get("stage") != "confirm" or row.get("phase") != "validation" or row.get("base") != "HPA":
            continue
        key = (row["source"], str(row["app"]), row["scenario"])
        if row.get("role") == "oracle_candidate":
            grouped.setdefault(key, []).append(row)
        elif row.get("role") == "oracle_locked":
            if key in locked:
                raise Incomplete("Duplicate known-delay validation lock")
            locked[key] = row
    accepted, grid = set(), candidate_grid(config)["HPA"]
    for source in SOURCES:
        for app in expected[source]:
            for case in PRIMARY:
                key = source, str(app), case
                group, lock = grouped.get(key, []), locked.get(key)
                if len(group) != 30 or sorted(int(r["candidate"]) for r in group) != list(range(30)) or lock is None:
                    continue
                if any(r["status"] != "OK" for r in group) or lock["status"] != "OK":
                    continue
                for row in group:
                    n, loss, paid = map(numeric, (row["minutes"], row["violations"], row["paid"]))
                    if not (math.isfinite(n) and n > 0 and math.isfinite(loss) and 0 <= loss <= n
                            and math.isfinite(paid) and paid > 0):
                        raise Incomplete("Invalid known-delay validation loss/cost")
                    cfg = grid[int(row["candidate"])]
                    if any(numeric(row[field]) != cfg[key] for field, key in (
                            ("config_risk", "risk"), ("parameter", "parameter"), ("window", "window"), ("stable", "stable"))):
                        raise Incomplete("Known-delay validation candidate differs from the frozen HPA grid")
                points = [dict(status=r["status"], candidate=int(r["candidate"]),
                    risk=numeric(r["violations"]) / numeric(r["minutes"]),
                    cost=numeric(r["paid"]) / numeric(r["minutes"])) for r in group]
                chosen = choose(points, config)
                selected = next(r for r in group if int(r["candidate"]) == chosen)
                fields = ("config_risk", "parameter", "window", "stable", "minutes", "violations", "paid")
                if chosen == int(lock["candidate"]) and all(numeric(selected[k]) == numeric(lock[k]) for k in fields):
                    accepted.add(key)
    return accepted, locked


def aggregate_episodes(rows, expected, config, eta=.1):
    """All 13 cases remain present; flash repeats yield one app/case observation."""
    grouped, flags, durations = {}, {}, {}
    for row in rows:
        if row.get("stage") != "confirm" or row.get("phase") != "test" or row.get("role") not in ("locked", "oracle_locked"):
            continue
        name = controller(row, eta)
        if name is None:
            continue
        source, app, case, rep = row["source"], str(row["app"]), row["scenario"], int(row["replicate"])
        if source not in expected or app not in set(map(str, expected[source])) or case not in CASES:
            raise Incomplete("Unexpected source/app/case in confirmation description")
        flag = int(row["contract"])
        key = source, app, case, rep
        if flag not in (0, 1) or key in flags and flags[key] != flag:
            raise Incomplete("Descriptive contract flag depends on controller")
        flags[key] = flag
        duration = numeric(row["minutes"])
        if not math.isfinite(duration) or duration <= 0 or key in durations and durations[key] != duration:
            raise Incomplete("Descriptive episode durations differ")
        durations[key] = duration
        grouped.setdefault((source, app, case, name), []).append(row)
    result = {}
    for source in SOURCES:
        for app in map(str, expected[source]):
            for case in CASES:
                reps = tuple(config["design"]["seeds"]) if case.startswith("flash") else (0,)
                keys = [(source, app, case, rep) for rep in reps]
                contract = int(all(flags[key] for key in keys)) if all(key in flags for key in keys) else None
                names = CONTROLLERS + ((ORACLE,) if case in PRIMARY and config["execution"].get("known_delay_hpa") else ())
                for name in names:
                    group = grouped.get((source, app, case, name), [])
                    status = "OK"
                    if len(group) != len(reps) or sorted(int(r["replicate"]) for r in group) != sorted(reps):
                        status = "MISSING_OR_DUPLICATE_RUN"
                    elif all(r["status"] == r.get("validation_status") == "NO_VALID_CONFIG" for r in group):
                        status = "NO_VALID_CONFIG"
                    elif any(r["status"] != "OK" for r in group):
                        status = "FAILED_RUN"
                    risk = cost = paid = math.nan
                    if status == "OK":
                        for row in group:
                            n, loss, bill = map(numeric, (row["minutes"], row["violations"], row["paid"]))
                            if not (math.isfinite(loss) and 0 <= loss <= n and math.isfinite(bill) and bill > 0):
                                raise Incomplete("Invalid observed descriptive loss/cost")
                        risk = float(np.mean([numeric(r["violations"]) / numeric(r["minutes"]) for r in group]))
                        cost = float(np.mean([numeric(r["paid"]) / numeric(r["minutes"]) for r in group]))
                        paid = sum(numeric(r["paid"]) for r in group)
                    result[source, app, case, name] = dict(status=status, contract=contract, risk=risk, cost=cost,
                        paid=paid, group=group, repeats=len(reps))
    return result


def all_case_description(episodes, expected):
    output = []
    for level, scope, sources in scopes():
        for case in CASES:
            for name in CONTROLLERS:
                entries = [(source, str(app), episodes[source, str(app), case, name]) for source in sources for app in expected[source]]
                for population in ("all_selected", "contract", "outside_contract"):
                    subset = [(s, a, r) for s, a, r in entries if population == "all_selected" or r["contract"] == int(population == "contract")]
                    unknown = sum(r["contract"] is None for _, _, r in entries)
                    unavailable = sum(r["status"] == "NO_VALID_CONFIG" for _, _, r in subset)
                    failed = sum(r["status"] not in ("OK", "NO_VALID_CONFIG") for _, _, r in subset)
                    observed = [r for _, _, r in subset if r["status"] == "OK"]
                    status = "INCOMPLETE" if failed or unknown else "OK" if observed else "EMPTY_SUPPORT"
                    output.append(dict(section="all_13_cases", scope_level=level, scope=scope, case=case,
                        contract_scope=population, controller=name, eta=.1 if name.endswith("_wrapped") else None,
                        status=status, selected_cells=len(subset), N_app_case_cells=len(observed),
                        distinct_apps=len(observed), unavailable_cells=unavailable, failed_cells=failed,
                        contract_unknown_cells=unknown,
                        risk_mean=float(np.mean([r["risk"] for r in observed])) if status == "OK" else math.nan,
                        cost_mean=float(np.mean([r["cost"] for r in observed])) if status == "OK" else math.nan,
                        compliance=float(np.mean([r["risk"] <= .06 + 1e-12 for r in observed])) if status == "OK" else math.nan,
                        pooling="unweighted_apps_flash_mean_then_contract_AND"))
    return output


def known_delay_description(rows, episodes, expected, config):
    accepted, validation = oracle_audit(rows, expected, config)
    names, output = ("HPA_raw", "HPA_wrapped", ORACLE), []
    for level, scope, sources in scopes():
        for case in (*PRIMARY, "PRIMARY_POOLED"):
            cases = PRIMARY if case == "PRIMARY_POOLED" else (case,)
            paired, unavailable, failed = [], 0, 0
            for source in sources:
                for app in map(str, expected[source]):
                    for actual_case in cases:
                        group = [episodes[source, app, actual_case, name] for name in names]
                        if any(r["contract"] is None for r in group) or any(r["status"] not in ("OK", "NO_VALID_CONFIG") for r in group):
                            failed += 1
                            continue
                        if not group[0]["contract"]:
                            continue
                        if any(r["status"] == "NO_VALID_CONFIG" for r in group):
                            unavailable += 1
                            continue
                        key = source, app, actual_case
                        test = group[2]["group"][0]
                        lock = validation.get(key, {})
                        signature = ("candidate", "config_risk", "parameter", "window", "stable")
                        if key not in accepted or any(numeric(test.get(k)) != numeric(lock.get(k)) for k in signature):
                            failed += 1
                            continue
                        paired.append((source, app, group))
            status = "INCOMPLETE" if failed else "OK" if paired else "EMPTY_PAIRED_SUPPORT"
            for index, name in enumerate(names):
                entries = [group[index] for _, _, group in paired]
                paid = sum(r["paid"] for r in entries)
                nominal = sum(group[0]["paid"] for _, _, group in paired)
                oracle = sum(group[2]["paid"] for _, _, group in paired)
                output.append(dict(section="known_delay_HPA", scope_level=level, scope=scope, case=case,
                    contract_scope="common_contract_three_controller_pairs", controller=name,
                    eta=.1 if name == "HPA_wrapped" else None, status=status,
                    selected_cells=sum(len(expected[s]) for s in sources) * len(cases),
                    N_app_case_cells=len(paired), distinct_apps=len({(s, a) for s, a, _ in paired}),
                    unavailable_cells=unavailable, failed_cells=failed,
                    risk_mean=float(np.mean([r["risk"] for r in entries])) if status == "OK" else math.nan,
                    cost_mean=float(np.mean([r["cost"] for r in entries])) if status == "OK" else math.nan,
                    compliance=float(np.mean([r["risk"] <= .06 + 1e-12 for r in entries])) if status == "OK" else math.nan,
                    paid_ratio_to_nominal=paid / nominal if status == "OK" else math.nan,
                    paid_ratio_to_known_delay=paid / oracle if status == "OK" else math.nan,
                    pooling="unweighted_paired_app_case_actual_paid_sums"))
    return output


def eta_case_description(rows, expected, config):
    """C: all five eta x 13 cases, using fixed source/vendor macro weights.

    Each source averages paired applications after averaging flash repetitions.
    Empty sources keep their fixed weight and yield NA; no reweighting or selection.
    """
    output = []
    weighted_scopes = [("source", source, {source: 1.}) for source in SOURCES] + [
        ("vendor", "Huawei", {SOURCES[0]: .5, SOURCES[1]: .5}),
        ("vendor", "Azure", {SOURCES[2]: 1.}),
        ("overall", "overall", dict(zip(SOURCES, (.25, .25, .5))))]
    metrics = ("risk_mean", "cost_mean", "compliance", "paired_cost_ratio_median")
    counts = ("selected_cells", "N_app_case_cells", "distinct_apps", "unavailable_cells", "failed_cells", "contract_unknown_cells")
    for eta in config["search"]["wrapper_eta"]:
        episodes = aggregate_episodes(rows, expected, config, eta)
        for case in CASES:
            for base in BASES:
                for population in ("all_selected", "contract", "outside_contract"):
                    points = {}
                    for source in SOURCES:
                        pairs, selected, unavailable, failed, unknown = [], 0, 0, 0, 0
                        for app in map(str, expected[source]):
                            pair = [episodes[source, app, case, f"{base}_{kind}"] for kind in ("raw", "wrapped")]
                            if pair[0]["contract"] is None:
                                unknown += 1
                                continue
                            if population != "all_selected" and pair[0]["contract"] != int(population == "contract"):
                                continue
                            selected += 1
                            if all(r["status"] == "NO_VALID_CONFIG" for r in pair):
                                unavailable += 1
                            elif any(r["status"] != "OK" for r in pair):
                                failed += 1
                            else:
                                pairs.append(pair)
                        status = "INCOMPLETE" if failed or unknown else "OK" if pairs else "EMPTY_SUPPORT"
                        for w in (0, 1):
                            points[source, w] = dict(status=status, selected_cells=selected,
                                N_app_case_cells=len(pairs), distinct_apps=len(pairs), unavailable_cells=unavailable,
                                failed_cells=failed, contract_unknown_cells=unknown,
                                risk_mean=float(np.mean([pair[w]["risk"] for pair in pairs])) if status == "OK" else math.nan,
                                cost_mean=float(np.mean([pair[w]["cost"] for pair in pairs])) if status == "OK" else math.nan,
                                compliance=float(np.mean([pair[w]["risk"] <= .06 + 1e-12 for pair in pairs])) if status == "OK" else math.nan,
                                paired_cost_ratio_median=float(np.median([pair[w]["cost"] / pair[0]["cost"] for pair in pairs])) if status == "OK" else math.nan)
                    for level, scope, weights in weighted_scopes:
                        for w, kind in enumerate(("raw", "wrapped")):
                            source_points = {source: points[source, w] for source in weights}
                            statuses = {point["status"] for point in source_points.values()}
                            status = "INCOMPLETE" if "INCOMPLETE" in statuses else "EMPTY_SUPPORT" if "EMPTY_SUPPORT" in statuses else "OK"
                            output.append(dict(section="eta_case_sensitivity", scope_level=level, scope=scope,
                                case=case, contract_scope=population, base=base, controller=f"{base}_{kind}",
                                eta=eta, status=status, frozen_primary_eta=int(math.isclose(eta, .1, abs_tol=1e-12)),
                                role="descriptive_only_no_reselection", pooling="fixed_source_vendor_macro_flash_app_mean",
                                **{name: sum(point[name] for point in source_points.values()) for name in counts},
                                **{name: sum(weights[source] * point[name] for source, point in source_points.items())
                                   if status == "OK" else math.nan for name in metrics}))
    return output


def confirmation_descriptions(rows, expected, config):
    if not math.isclose(config["frozen_numeric"]["selected_eta"], .1, rel_tol=0, abs_tol=1e-12):
        raise Incomplete("This confirmation freezes primary eta=0.1")
    primary = panels(rows, expected, .1, "confirm")
    risk = risk_distribution(primary, config)
    episodes = aggregate_episodes(rows, expected, config)
    described = all_case_description(episodes, expected)
    if config["execution"].get("known_delay_hpa"):
        described += known_delay_description(rows, episodes, expected, config)
    if config["execution"].get("confirm_eta_sensitivity"):
        described += eta_case_description(rows, expected, config)
        datasets = {eta: panels(rows, expected, eta, "confirm") for eta in config["search"]["wrapper_eta"]}
        for row in exploration_rows(datasets, config, .1):
            if row["metric"] in ("selection_eligible", "selection_reason", "selected", "grid_boundary_limitation"):
                if row["metric"] != "selected":
                    continue
                row = dict(row, metric="frozen_primary_eta")
            described.append(dict(section="eta_sensitivity", case="PRIMARY_POOLED", role="descriptive_only_no_reselection",
                                   pooling="primary_macro_estimator_points_only", **row))
    return risk, described


def write_confirmation(rows, expected, config, reference):
    risk, described = confirmation_descriptions(rows, expected, config)
    for name, records in (("v2-lite-confirm-risk.csv", risk), ("v2-lite-confirm-descriptive.csv", described)):
        fields = ["stage", "config_ref", *dict.fromkeys(key for row in records for key in row)]
        with (ROOT / "results" / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            for row in records:
                writer.writerow({key: "NA" if value is None or isinstance(value, (float, np.floating)) and not math.isfinite(value) else value
                                 for key, value in dict(stage="confirm", config_ref=reference, **row).items()})


def main():
    config = load_config()  # Canonical configuration only; no --config override.
    reference = require_stage_gate(config, "confirm")
    expected = validate_manifest(config, "confirm")
    with (ROOT / config["io"]["summary_csv"]).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if any(row["config_ref"] != reference for row in rows if row["stage"] == "confirm"):
        raise Incomplete("Confirmation descriptions require their frozen reference")
    write_confirmation(rows, expected, config, reference)


if __name__ == "__main__":
    main()
