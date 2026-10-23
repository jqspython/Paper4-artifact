"""v2-lite persistence pipeline. Real-data IO is behind reviewed freeze gates."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import datetime as dt
import hashlib
import math
from pathlib import Path
import sqlite3
import subprocess
import tomllib

import numpy as np

from cdt_pilot import BACCLite, safety_constants, service_lag
from gate2_runner import configurations, scenarios, transform
from gate2_wrapper import (BACCCal, FeedbackWrapper, hpa_request,
                           pac_requests, point_margin_request)
from paper4 import Actuator
from scripts.analyze_gate2 import choose


ROOT = Path(__file__).resolve().parent
BASES = ("HPA", "Point+margin", "PAC-h", "BACC-cal")
COMPETITOR = "BACC-lite"
SOURCES = ("huawei_private2023", "huawei_public2023", "azure2019")
PRIMARY = ("delay120", "delay180", "delay300", "delay600")
CONFIRM_COUNTS = dict(zip(SOURCES, (57, 80, 80)))
FIELDS = ("stage config_ref source app scenario replicate phase role base wrapped candidate eta epsilon "
          "config_risk parameter window stable status validation_status violations minutes paid ready actions "
          "contract physical_floor lambda_final lambda_min cdt_bound capacity minimum maximum rate "
          "h_actual h_assumed dmax lambda_safe L K").split()
CRITICAL = ("configs/v2-lite.toml", "configs/gate2.toml", "docs/v2-protocol.md", "v2_lite.py", "gate2_wrapper.py",
            "paper4.py", "cdt_pilot.py", "gate2_runner.py", "gate2_kernel.py", "gate2_pac.py",
            "scripts/analyze_gate2.py", "tests/test_v2_lite.py", "tests/test_gate2_wrapper.py",
            "scripts/analyze_v2_lite.py", "tests/test_v2_lite_analysis.py", "scripts/build_paper_artifacts.py",
            "scripts/benchmark_v2_lite.py", "results/v2-lite-workload.csv",
            "docs/v2-lite-pipeline.md", "docs/v2-lite-reuse-audit.md", "docs/v2-design-review.md",
            "scripts/audit_v2_lite_reuse.py", "results/v2-lite-lock-reuse.csv",
            "data/processed/gate2/selected-unit-ids.csv",
            "data/processed/gate2/eligible-unit-pool.csv",
            "scripts/prepare_gate2_data.py", "scripts/prepare_v2_lite_confirmation.py",
            "tests/test_v2_lite_confirmation_data.py", "scripts/report_v2_lite_confirmation.py",
            "scripts/plan_v2_confirmation.py", "results/v2-confirm-workload.csv")


def load_config(path=None):
    return tomllib.loads(Path(path or ROOT / "configs/v2-lite.toml").read_text(encoding="utf-8"))


def candidate_grid(config):
    old = configurations(config)
    result = {name: old[name] for name in (*BASES[:3], COMPETITOR)}
    search = config["search"]
    if len(search["risk_levels"]) != 6 or len(search["bacc_cal_window"]) != 5:
        raise ValueError("BACC-cal requires six alpha values crossed with five windows")
    result["BACC-cal"] = [dict(risk=r, parameter=0., window=w, stable=300)
                          for r in search["risk_levels"] for w in search["bacc_cal_window"]]
    for name, candidates in result.items():
        if len(candidates) != 30 or len({tuple(sorted(c.items())) for c in candidates}) != 30:
            raise ValueError(f"{name} requires 30 distinct, ordered configurations")
    return result


def select_nominal(rows, config, methods=BASES):
    """One application: v1 choose on observed risk/cost, never candidate alpha."""
    rows = list(rows)
    for field in ("trace", "source", "app", "predictor", "seed"):
        if len({r[field] for r in rows if field in r}) > 1:
            raise ValueError(f"Selection mixes {field}")
    for row in rows:
        if (row.get("phase") != "validation" or row.get("scenario") != "nominal"
                or row.get("budget_mode", "causal") != "causal"
                or row.get("predictor", "persistence") != "persistence"
                or int(row.get("seed", 0)) != 0 or row.get("wrapped", 0)):
            raise ValueError("Selection requires unwrapped persistence nominal validation")
    grid, result = candidate_grid(config), {}
    for method in methods:
        if method not in grid:
            raise ValueError(f"Unknown base: {method}")
        group = [r for r in rows if r.get("base", r.get("method")) == method]
        indices = [r["candidate"] for r in group]
        if len(indices) != len(set(indices)):
            raise ValueError("Duplicate nominal candidate")
        for row in group:
            index = row["candidate"]
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 30:
                raise ValueError("Candidate index is outside the fixed grid")
            if "config" in row and row["config"] != grid[method][index]:
                raise ValueError("Candidate configuration differs from the fixed grid")
            if row["status"] == "OK" and not (math.isfinite(row["risk"]) and 0 <= row["risk"] <= 1
                                                 and math.isfinite(row["cost"]) and row["cost"] >= 0):
                raise ValueError("Invalid observed validation risk/cost")
        index = choose(group, config)
        point = next((r for r in group if r["candidate"] == index), None)
        cutoff = config["gate"]["budget"] + config["gate"]["budget_tolerance"] + 1e-12
        result[method] = dict(candidate=index, config=dict(grid[method][index]) if index is not None else None,
                              validation_status="NO_VALID_CONFIG" if point is None else
                              "OK" if point["risk"] <= cutoff else "VALIDATION_BUDGET_NOT_MET")
    return result


def validate_config(config):
    design = config["design"]
    if (tuple(design["traces"]) != SOURCES or design["predictors"] != ["persistence"]
            or design["primary_predictor"] != "persistence" or design["predictor_seed"] != 0):
        raise ValueError("v2-lite requires the three specified sources and persistence seed 0")
    execution = dict(config.get("execution", {}))
    descriptions = [execution.pop(key, False) for key in ("confirm_eta_sensitivity", "known_delay_hpa")]
    if (any(type(value) is not bool for value in descriptions)
            or isinstance(design["workers"], bool) or not isinstance(design["workers"], int)
            or not 1 <= design["workers"] <= 8 or execution != dict(exploration_scope="primary",
            confirmation_scope="all_descriptive", engine="reference_python")):
        raise ValueError("Use 1..8 reference-Python workers and the declared stage scopes")
    if tuple(config["stress"]["primary"]) != PRIMARY:
        raise ValueError("Primary selection is restricted to the four delay cases")
    if design["seeds"] != [17, 29, 43, 71, 101]:
        raise ValueError("The five fixed flash-position repetitions are required")
    if config["gate"]["budget"] != .05 or config["gate"]["budget_tolerance"] != .001:
        raise ValueError("The nominal selection budget must remain .051")
    if config["search"]["wrapper_eta"] != [.02, .05, .1, .2, .5]:
        raise ValueError("The user-specified five eta candidates are required")
    if config["plant"]["step_seconds"] != 60 or config["plant"]["capacity_fit_peak_divisor"] != 8:
        raise ValueError("The minute clock and fit-peak capacity divisor are fixed")
    candidate_grid(config)


def execution_cases(config, stage):
    """Exploration selects eta on four delays; confirmation retains all 13 cases."""
    if config.get("execution", {}).get("exploration_scope") != "primary":
        raise ValueError("Exploration execution_scope must be explicitly primary")
    if stage not in ("explore", "confirm"):
        raise ValueError("Unknown execution stage")
    return [case for case in scenarios(config)
            if stage == "confirm" or case["name"] in config["stress"]["primary"]]


def plan(config):
    """Counts only: neither source arrays nor a database are opened."""
    validate_config(config)
    cases = scenarios(config)
    episodes = sum(5 if c["transform"] == "flash" else 1 for c in cases)
    exploration = execution_cases(config, "explore")
    explore_episodes = sum(5 if c["transform"] == "flash" else 1 for c in exploration)
    confirm_etas = len(config["search"]["wrapper_eta"]) if config["execution"].get("confirm_eta_sensitivity", False) else 1
    confirm_tests = episodes * (5 + 4 * confirm_etas)
    oracle_slots = len(PRIMARY) * (30 + 1 + 1) if config["execution"].get("known_delay_hpa", False) else 0
    confirm_slots = 150 + 5 + confirm_tests + oracle_slots
    return dict(sources=list(SOURCES), vendors=2, predictor="persistence", predictor_seed=0,
                workers=config["design"]["workers"], database_writers=1,
                candidate_slots={name: len(grid) for name, grid in candidate_grid(config).items()},
                primary=list(PRIMARY), descriptive_cases=[c["name"] for c in cases],
                case_repetitions=episodes, exploration_cases=[c["name"] for c in exploration],
                exploration_case_repetitions=explore_episodes, nominal_candidate_slots_per_app=150,
                explore_locked_slots_per_app=explore_episodes * (5 + 4 * 5),
                confirm_locked_slots_per_app=confirm_tests,
                confirm_wrapper_eta_count=confirm_etas, confirm_oracle_slots_per_app=oracle_slots,
                confirm_nominal_slots_per_app=150 + 5, confirm_total_slots_per_app=confirm_slots,
                confirm_total_slots_expected=confirm_slots * sum(CONFIRM_COUNTS.values()),
                exploration_applications=3 * config["design"]["exploration_apps_per_trace"],
                confirmation_applications_expected=sum(CONFIRM_COUNTS.values()),
                cached_reuse="conditional on audited CSV, exact configuration and reviewed stage")


def primary_rows(rows, config):
    """Descriptive cases are retained in outputs, excluded from eta endpoints."""
    return [r for r in rows if r.get("phase") == "test" and r.get("role") == "locked"
            and r["scenario"] in config["stress"]["primary"]]


def require_stage_gate(config, stage, *, experiment=False):
    if stage not in ("explore", "confirm"):
        raise ValueError("A real-data stage must be explore or confirm")
    gates = config["gates"]
    reviews = ["design_human_review"]
    if stage == "confirm":
        reviews += ["confirm_human_review", "confirm_numeric_review"]
        if config["frozen_numeric"]["selected_eta"] not in config["search"]["wrapper_eta"]:
            raise RuntimeError("Confirm eta is UNSELECTED or outside the frozen candidate set")
    if any(gates.get(key) != "APPROVED" for key in reviews):
        raise RuntimeError("Human/numeric review is not APPROVED; no real-data IO is permitted")
    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=ROOT, encoding="utf-8")
    try:
        git("rev-parse", "--verify", gates["design_tag"] + "^{commit}")
        tag = gates["design_tag"] if stage == "explore" else gates["confirm_tag"]
        reference = git("rev-parse", "--verify", tag + "^{commit}").strip()
        for name in CRITICAL:
            git("cat-file", "-e", f"{reference}:{name}")
            expected_blob = git("rev-parse", f"{reference}:{name}").strip()
            actual_blob = git("hash-object", f"--path={name}", name).strip()
            if actual_blob != expected_blob:
                raise RuntimeError(f"Frozen source differs: {name}")
    except subprocess.CalledProcessError as error:
        raise RuntimeError("The required freeze reference/source is missing") from error
    deadline = dt.datetime.fromisoformat(config["design"]["deadline"])
    if experiment and dt.datetime.now(dt.timezone.utc) > deadline:
        raise RuntimeError("The experiment deadline has passed")
    return reference


def prepare_episode(demand, fit_end, start, end, case, config, replicate=0):
    demand = np.asarray(demand, dtype=np.float64)
    if (demand.ndim != 1 or not np.all(np.isfinite(demand)) or np.any(demand < 0)
            or not 0 < fit_end <= start < end <= len(demand)):
        raise ValueError("Invalid causal demand/split array")
    changed = transform(demand, start, end, case, replicate)
    plant = config["plant"]
    capacity = float(np.max(demand[:fit_end])) / plant["capacity_fit_peak_divisor"]
    if capacity <= 0:
        raise ValueError("Positive fit peak is required")
    actual, assumed = service_lag(case["delay"]), service_lag(plant["assumed_startup_seconds"])
    dmax = service_lag(plant["contract_startup_seconds"])
    floor = float(np.mean(changed[start:end] > capacity * case["maximum"]))
    return dict(demand=changed, fit_end=fit_end, start=start, end=end, capacity=capacity,
                case=case, replicate=replicate, h_actual=actual, h_assumed=assumed, dmax=dmax,
                physical_floor=floor, contract=int(floor == 0 and actual <= dmax))


def calibration_scores(episode, window):
    start, h = episode["start"], episode["h_assumed"]
    y = episode["demand"]
    targets = np.arange(max(episode["fit_end"] + h, start - window), start)
    if not len(targets):
        raise ValueError("No mature preceding calibration labels")
    return np.clip((y[targets] + 1) / (y[targets - h] + 1), .05, 20.)


def proposal_path(episode, base, cfg, config):
    """No actuator/loss input: compute once and reuse for all wrapper eta values."""
    y, start, end, c = (episode[key] for key in ("demand", "start", "end", "capacity"))
    if base == "PAC-h":
        return pac_requests(y, y, c, episode["h_assumed"], cfg["window"], cfg["risk"],
                            config["search"]["pac_confidence_failure"], start=start)
    if base == "BACC-cal":
        cal = BACCCal(c, cfg["risk"], episode["h_assumed"], cfg["window"],
                      calibration_scores(episode, cfg["window"]))
        return np.asarray([cal.suggest(t, float(value), float(value))
                           for t, value in enumerate(y[start:end])], dtype=float)
    function = hpa_request if base == "HPA" else point_margin_request if base == "Point+margin" else None
    if function is None:
        raise ValueError("Full BACC-lite is not a loss-independent proposal path")
    return np.asarray([function(float(value), c, cfg["parameter"]) for value in y[start:end]], dtype=float)


def evaluate_episode(episode, base, cfg, config, eta=None, proposals=None):
    """Pure array evaluation. Its caller is responsible for reviewed real-data IO."""
    if base not in (*BASES, COMPETITOR) or (base == COMPETITOR and eta is not None):
        raise ValueError("Unknown base or attempted wrapping of full BACC-lite")
    if eta is not None and eta not in config["search"]["wrapper_eta"]:
        raise ValueError("eta is outside the frozen candidates")
    settings, case = config["plant"], episode["case"]
    length = episode["end"] - episode["start"]
    row = dict(scenario=case["name"], replicate=episode["replicate"], base=base, wrapped=int(eta is not None),
               eta=eta, epsilon=config["gate"]["budget"], config=dict(cfg), config_risk=cfg["risk"],
               parameter=cfg["parameter"], window=cfg["window"], stable=cfg["stable"], status="OK",
               minutes=length, contract=episode["contract"], physical_floor=episode["physical_floor"],
               capacity=episode["capacity"], minimum=settings["minimum"], maximum=case["maximum"],
               rate=case["rate"], h_actual=episode["h_actual"], h_assumed=episode["h_assumed"],
               dmax=episode["dmax"], lambda_final=None, lambda_min=None, cdt_bound=None,
               lambda_safe=None, L=None, K=None)
    for key in ("violations", "paid", "ready", "actions", "risk", "cost"):
        row[key] = None
    if base != COMPETITOR:
        try:
            proposals = proposal_path(episode, base, cfg, config) if proposals is None else np.asarray(proposals)
        except ValueError as error:
            if base == "PAC-h" and ("No finite PAC rank" in str(error) or "Saturated PAC rank" in str(error)):
                row["status"] = "NO_RANK"
                return row
            raise
        if proposals.shape != (length,) or not np.all(np.isfinite(proposals)) or np.any(proposals < 0):
            raise ValueError("Base proposals must be finite, nonnegative and cover every tick")
    actuator = Actuator(step_seconds=settings["step_seconds"],
                        delay_seconds=case["delay"] + settings["step_seconds"],
                        capacity=episode["capacity"], minimum=settings["minimum"],
                        maximum=case["maximum"], max_change=case["rate"], stable_seconds=cfg["stable"])
    wrapped = FeedbackWrapper(actuator, row["epsilon"], eta) if eta is not None else None
    full = BACCLite(episode["capacity"], cfg["risk"], episode["h_assumed"], length,
                    calibration_scores(episode, cfg["window"]), cfg["window"], cfg["parameter"]) if base == COMPETITOR else None
    violations = paid = ready_cost = actions = 0
    minimum_lambda = 0.
    for t, value in enumerate(episode["demand"][episode["start"]:episode["end"]]):
        before = actuator.replicas + sum(delta for _, delta in actuator.pending)
        if wrapped is not None:
            step = wrapped.step(t, float(value), float(proposals[t]))
            loss, ready, charged = step["loss"], step["ready"], step["paid"]
            minimum_lambda = min(minimum_lambda, wrapped.value)
        else:
            loss = int(value > actuator.begin(t))
            ready = actuator.replicas
            charged = ready + sum(max(0, delta) for _, delta in actuator.pending)
            request = full.decide(t, float(value), float(value), loss)[0] if full else float(proposals[t])
            actuator.command(t, request)
        after = actuator.replicas + sum(delta for _, delta in actuator.pending)
        actions += int(before != after)
        violations += loss
        paid += charged
        ready_cost += ready
    row.update(violations=violations, paid=paid, ready=ready_cost, actions=actions,
               risk=violations / length, cost=paid / length)
    if wrapped is not None:
        safe, settling, window = safety_constants(case["maximum"], settings["minimum"],
                                                 case["rate"], episode["dmax"], row["epsilon"])
        row.update(lambda_final=wrapped.value, lambda_min=minimum_lambda, lambda_safe=safe, L=settling, K=window)
        if not math.isclose(row["risk"], row["epsilon"] - wrapped.value / (eta * length), abs_tol=1e-10):
            raise AssertionError("Actual-loss recursion identity differs")
        if row["contract"]:
            row["cdt_bound"] = min(1., row["epsilon"] + (-safe / eta + window) / length)
            if minimum_lambda < safe - window * eta - 1e-8 or row["risk"] > row["cdt_bound"] + 1e-10:
                raise AssertionError("Conditional dynamic bound differs")
    return row


def observed_point(row):
    return dict(row, method=row["base"], predictor="persistence", seed=0, budget_mode="causal")


def known_delay_hpa_pipeline(demand, fit_end, cal_end, val_end, config):
    """Descriptive oracle: known actuator delay, chronological nominal validation only."""
    for case in execution_cases(config, "explore"):
        validation = prepare_episode(demand, fit_end, cal_end, val_end, case, config)
        points = []
        for candidate, cfg in enumerate(candidate_grid(config)["HPA"]):
            point = evaluate_episode(validation, "HPA", cfg, config)
            point.update(candidate=candidate, phase="validation", role="oracle_candidate")
            points.append(point)
            yield point
        # The signal is still nominal; only its actual actuator delay differs.
        inputs = [observed_point(dict(point, scenario="nominal")) for point in points]
        selection = select_nominal(inputs, config, methods=("HPA",))["HPA"]
        candidate, cfg = selection["candidate"], selection["config"]
        locked = next((point for point in points if point["candidate"] == candidate), None)
        if locked is not None:
            yield dict(locked, role="oracle_locked", validation_status=selection["validation_status"])
        else:
            yield dict(scenario=case["name"], replicate=0, phase="validation", role="oracle_locked",
                       base="HPA", wrapped=0, eta=None, candidate=None, status="NO_VALID_CONFIG",
                       validation_status="NO_VALID_CONFIG", minutes=val_end-cal_end,
                       contract=validation["contract"], physical_floor=validation["physical_floor"])
        test = prepare_episode(demand, fit_end, val_end, len(demand), case, config)
        if cfg is None:
            yield dict(scenario=case["name"], replicate=0, phase="test", role="oracle_locked",
                       base="HPA", wrapped=0, eta=None, candidate=None, status="NO_VALID_CONFIG",
                       validation_status="NO_VALID_CONFIG", minutes=len(demand)-val_end,
                       contract=test["contract"], physical_floor=test["physical_floor"])
        else:
            point = evaluate_episode(test, "HPA", cfg, config)
            yield dict(point, candidate=candidate, phase="test", role="oracle_locked",
                       validation_status=selection["validation_status"])


def application_pipeline(demand, fit_end, cal_end, val_end, config, stage="explore", reused=None):
    """Yield only array-derived rows; no files, IDs, fitting or parameter search beyond nominal."""
    if stage not in ("explore", "confirm"):
        raise ValueError("Unknown stage")
    if stage == "confirm" and config["frozen_numeric"]["selected_eta"] not in config["search"]["wrapper_eta"]:
        raise ValueError("Confirm eta is not selected")
    grid, selected = candidate_grid(config), {}
    nominal = next(c for c in scenarios(config) if c["name"] == "nominal")
    episode = prepare_episode(demand, fit_end, cal_end, val_end, nominal, config)
    for base in (*BASES, COMPETITOR):
        if reused and base in reused:
            point = dict(reused[base])
            if point["config"] != grid[base][point["candidate"]]:
                raise ValueError("Reused selected configuration differs from its candidate")
            points = [point]
        else:
            points = []
            for candidate, cfg in enumerate(grid[base]):
                point = evaluate_episode(episode, base, cfg, config)
                point.update(candidate=candidate, phase="validation", role="candidate")
                points.append(point)
                yield point
        inputs = [observed_point(dict(point, phase="validation", scenario="nominal")) for point in points]
        selection = select_nominal(inputs, config, methods=(base,))[base]
        selected[base] = selection
        locked = next((p for p in points if p["candidate"] == selection["candidate"]), None)
        if locked is not None:
            yield dict(locked, phase="validation", role="locked", validation_status=selection["validation_status"])
    etas = (config["search"]["wrapper_eta"] if stage == "explore" or
            config["execution"].get("confirm_eta_sensitivity", False) else [config["frozen_numeric"]["selected_eta"]])
    if any(eta not in config["search"]["wrapper_eta"] for eta in etas):
        raise ValueError("Confirm eta is not selected")
    for case in execution_cases(config, stage):
        repeats = config["design"]["seeds"] if case["transform"] == "flash" else [0]
        for repeat in repeats:
            episode = prepare_episode(demand, fit_end, val_end, len(demand), case, config, repeat)
            for base in (*BASES, COMPETITOR):
                selection = selected[base]
                candidate, cfg = selection["candidate"], selection["config"]
                if cfg is None:
                    for eta in ([None] + etas if base in BASES else [None]):
                        yield dict(scenario=case["name"], replicate=repeat, phase="test", role="locked", base=base,
                                   wrapped=int(eta is not None), eta=eta, candidate=None, status="NO_VALID_CONFIG",
                                   validation_status="NO_VALID_CONFIG", minutes=len(demand)-val_end,
                                   contract=episode["contract"], physical_floor=episode["physical_floor"])
                    continue
                proposals = None
                if base != COMPETITOR:
                    proposals = proposal_path(episode, base, cfg, config)
                for eta in ([None] + etas if base in BASES else [None]):
                    row = evaluate_episode(episode, base, cfg, config, eta, proposals)
                    row.update(candidate=candidate, phase="test", role="locked",
                               validation_status=selection["validation_status"])
                    yield row
    if stage == "confirm" and config["execution"].get("known_delay_hpa", False):
        yield from known_delay_hpa_pipeline(demand, fit_end, cal_end, val_end, config)


def load_reuse(config):
    """Only audited persistence nominal locks; invalid entries fall back to validation."""
    path = ROOT / config["io"]["reuse_csv"]
    if not path.is_file():
        return {}
    grid, result, seen, duplicates = candidate_grid(config), {}, set(), set()
    old = tomllib.loads((ROOT / "configs/gate2.toml").read_text(encoding="utf-8"))
    old_grid = configurations(old)
    if (config["plant"] != old["plant"]
            or config["search"]["pac_confidence_failure"] != old["search"]["pac_confidence_failure"]
            or any(config["design"][k] != old["design"][k] for k in ("fit_days", "calibration_days", "validation_days"))):
        return {}
    compatible = {base for base in BASES[:3] if grid[base] == old_grid[base]}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for old in csv.DictReader(stream):
            key = (old.get("trace"), old.get("app"), old.get("method"))
            if key in seen:
                duplicates.add(key)
                result.pop(key, None)
                continue
            seen.add(key)
            try:
                base, candidate = old["method"], int(old["candidate"])
                cfg = {k: float(old[k]) for k in ("risk", "parameter", "window", "stable")}
                checks = (old["audit_status"] == "PASS" and old["reuse_eligible"] == "1"
                          and old["predictor"] == "persistence" and old["seed"] == "0"
                          and old["phase"] == "validation" and old["scenario"] == "nominal"
                          and old["budget_mode"] == "causal" and old["status"] == "OK"
                          and base in compatible and 0 <= candidate < 30 and int(old["candidate_slots"]) == 30
                          and all(old[k] in ("1", "True") for k in ("candidate_match", "validation_match", "config_match"))
                          and math.isclose(float(old["validation_cutoff"]), .051, abs_tol=1e-12, rel_tol=0)
                          and all(int(old[k]) == candidate for k in ("reference_candidate", "v1_candidate", "v2_candidate"))
                          and cfg == grid[base][candidate]
                          and all(math.isclose(float(old[k]), float(old["locked_" + k]), abs_tol=1e-12, rel_tol=0)
                                  for k in ("validation_risk", "validation_cost")))
                risk, cost = float(old["validation_risk"]), float(old["validation_cost"])
                if checks and 0 <= risk <= 1 and math.isfinite(cost) and cost >= 0:
                    result[key] = dict(base=base, candidate=candidate, config=dict(grid[base][candidate]),
                                       status="OK", risk=risk, cost=cost)
            except (ValueError, KeyError, IndexError):
                continue
    for key in duplicates:
        result.pop(key, None)
    return result


def validate_manifest(config, stage, directory=None):
    """Identity metadata only. Validate prepared IDs; never select/write a cohort."""
    if stage not in ("explore", "confirm"):
        raise ValueError("Unknown identity stage")
    registry = ROOT / config["io"]["exploration_data"]
    directory = Path(directory) if directory is not None else ROOT / config["io"][
        "exploration_data" if stage == "explore" else "confirmation_data"]
    manifest = directory / "selected-unit-ids.csv"
    if not manifest.is_file():
        raise RuntimeError("NOT_PREPARED: an independently prepared identity manifest is required")
    def read_ids(path):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        pairs = [(r.get("source", r.get("trace")), r["unit_id"]) for r in rows]
        if len(pairs) != len(set(pairs)) or {s for s, _ in pairs} != set(SOURCES):
            raise ValueError("Identity registry has duplicates or a missing/unknown source")
        return {source: tuple(app for s, app in pairs if s == source) for source in SOURCES}
    prepared, old = read_ids(manifest), read_ids(registry / "selected-unit-ids.csv")
    if any(len(ids) != config["design"]["exploration_apps_per_trace"] for ids in old.values()):
        raise ValueError("Original exploration identity counts differ")
    if stage == "explore":
        expected = old
    else:
        pool = read_ids(registry / "eligible-unit-pool.csv")
        prefix = config["selection"]["hash_prefix"]
        expected = {source: tuple(sorted(set(pool[source]) - set(old[source]), key=lambda app: (
            hashlib.sha256((prefix + app).encode()).digest(), app))[:config["design"]["confirmation_cap_per_trace"]])
                    for source in SOURCES}
        if any(len(expected[source]) != CONFIRM_COUNTS[source] for source in SOURCES):
            raise ValueError("Frozen eligible frame does not support the fixed 57/80/80 confirmation cohort")
    if any(set(prepared[source]) != set(expected[source]) for source in SOURCES):
        raise ValueError("Prepared identities differ from the frozen source-qualified hash/exclusion rule")
    return prepared


def load_source(directory, source, config, stage):
    path = Path(directory) / f"{source}.npz"
    if not path.is_file():
        raise RuntimeError(f"NOT_PREPARED: {path}")
    with np.load(path, allow_pickle=False) as data:
        result = {k: data[k] for k in ("unit_id", "fit_end", "cal_end", "val_end")}
    ids = [str(x) for x in result["unit_id"]]
    if len(ids) != len(set(ids)) or not len(ids):
        raise ValueError("Malformed source application identities/arrays")
    if stage == "explore" and len(ids) != config["design"]["exploration_apps_per_trace"]:
        raise ValueError("Exploration must contain the original 50 applications per source")
    if stage == "confirm" and len(ids) > config["design"]["confirmation_cap_per_trace"]:
        raise ValueError("Prepared confirmation IDs exceed the source cap; no truncation is allowed")
    boundaries = tuple(int(result[k]) for k in ("fit_end", "cal_end", "val_end"))
    if boundaries != (4320, 7200, 11520):
        raise ValueError("The fixed 3/2/3-day chronological splits differ")
    expected = validate_manifest(config, stage, directory)[source]
    if set(ids) != set(expected):
        raise ValueError("NPZ identities differ from the reviewed manifest")
    # Only approved identities reach the demand body; no ID is generated or written.
    with np.load(path, allow_pickle=False) as data:
        result["demand"] = data["demand"]
    y = result["demand"]
    if y.ndim != 2 or y.shape[0] != len(ids) or y.shape[1] <= boundaries[-1]:
        raise ValueError("Malformed demand matrix or test boundary")
    result["unit_id"] = ids
    return result


def output_database(path, stage, reference):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    types = {key: "REAL" for key in FIELDS}
    for key in ("stage", "config_ref", "source", "app", "scenario", "phase", "role", "base", "status", "validation_status"):
        types[key] = "TEXT"
    for key in ("replicate", "wrapped", "candidate", "window", "stable", "violations", "minutes", "paid", "ready",
                "actions", "contract", "minimum", "maximum", "rate", "h_actual", "h_assumed", "dmax", "lambda_safe", "L", "K"):
        types[key] = "INTEGER"
    db.execute("CREATE TABLE IF NOT EXISTS runs (" + ",".join(f'"{k}" {types[k]}' for k in FIELDS) + ")")
    db.execute("CREATE TABLE IF NOT EXISTS stage_refs (stage TEXT PRIMARY KEY,config_ref TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS jobs (stage TEXT,config_ref TEXT,source TEXT,app TEXT,PRIMARY KEY(stage,config_ref,source,app))")
    previous = db.execute("SELECT config_ref FROM stage_refs WHERE stage=?", (stage,)).fetchone()
    if previous is not None and previous[0] != reference:
        db.close()
        raise RuntimeError("Checkpoint uses a different frozen reference; refusing mixed-stage resumption")
    db.execute("INSERT OR IGNORE INTO stage_refs VALUES (?,?)", (stage, reference))
    db.commit()
    return db


def export_summary(db, output):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        writer.writerows(db.execute("SELECT " + ",".join('"' + k + '"' for k in FIELDS) +
                                   " FROM runs ORDER BY stage,source,app,phase,role,base,scenario,replicate,wrapped,eta"))


def application_task(task):
    """Spawn-safe one-application computation; no file IO or shared database."""
    source, app, demand, fit_end, cal_end, val_end, config, stage, reused = task
    return source, app, list(application_pipeline(demand, fit_end, cal_end, val_end, config, stage, reused))


def run_stage(config, stage):
    """The sole real-data entry point: gates precede every NPZ/CSV/SQLite access."""
    validate_config(config)
    reference = require_stage_gate(config, stage, experiment=True)
    directory = ROOT / config["io"]["exploration_data" if stage == "explore" else "confirmation_data"]
    if any(not (directory / f"{source}.npz").is_file() for source in SOURCES):
        raise RuntimeError("NOT_PREPARED: no source substitution, ID selection or data download is performed")
    reuse = load_reuse(config) if stage == "explore" else {}
    source_data = {source: load_source(directory, source, config, stage) for source in SOURCES}
    if stage == "confirm" and sum(len(data["unit_id"]) for data in source_data.values()) != 217:
        raise ValueError("Prepared confirmation cohort must contain the fixed 217 identities")
    db = output_database(ROOT / config["io"]["output_db"], stage, reference)
    completed = {tuple(row) for row in db.execute("SELECT source,app FROM jobs WHERE stage=? AND config_ref=?", (stage, reference))}
    def tasks():
        for source in SOURCES:
            data = source_data[source]
            for index, app in enumerate(data["unit_id"]):
                if (source, app) in completed:
                    continue
                reused = {base: reuse[source, app, base] for base in BASES[:3] if (source, app, base) in reuse}
                # Reused counts are reconstructed only when exact integer aggregates are recoverable.
                n = int(data["val_end"] - data["cal_end"])
                for base, point in list(reused.items()):
                    violations, paid = round(point["risk"] * n), round(point["cost"] * n)
                    if not (math.isclose(violations / n, point["risk"], abs_tol=1e-12, rel_tol=0)
                            and math.isclose(paid / n, point["cost"], abs_tol=1e-12, rel_tol=0)):
                        reused.pop(base)
                        continue
                    point.update(minutes=n, violations=violations, paid=paid, ready=None, actions=None,
                                 wrapped=0, eta=None, epsilon=config["gate"]["budget"],
                                 config_risk=point["config"]["risk"], parameter=point["config"]["parameter"],
                                 window=point["config"]["window"], stable=point["config"]["stable"])
                yield (source, app, data["demand"][index], int(data["fit_end"]), int(data["cal_end"]),
                       int(data["val_end"]), config, stage, reused)
    pool = None
    try:
        if config["design"]["workers"] == 1:
            results = map(application_task, tasks())
        else:
            pool = ProcessPoolExecutor(max_workers=config["design"]["workers"])
            results = pool.map(application_task, tasks(), chunksize=1)
        # Ordered map preserves source/NPZ order; only this process writes SQLite.
        for source, app, rows in results:
            with db:
                for row in rows:
                    row.update(stage=stage, config_ref=reference, source=source, app=app)
                    db.execute("INSERT INTO runs VALUES (" + ",".join("?" for _ in FIELDS) + ")",
                               [row.get(field) for field in FIELDS])
                db.execute("INSERT INTO jobs VALUES (?,?,?,?)", (stage, reference, source, app))
            print(f"{stage}: completed {source}/{app}", flush=True)
        export_summary(db, ROOT / config["io"]["summary_csv"])
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        db.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "self-check", "explore", "confirm"), default="plan")
    args = parser.parse_args(argv)
    config = load_config()
    if args.stage == "self-check":
        import unittest
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_v2_lite.py")
        if not suite.countTestCases():
            raise RuntimeError("The required synthetic properties are missing")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)
    elif args.stage == "plan":
        for key, value in plan(config).items():
            print(f"{key}: {value}")
    else:
        run_stage(config, args.stage)


if __name__ == "__main__":
    main()
