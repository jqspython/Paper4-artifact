"""Synthetic single-process timings plus workload arithmetic; no scientific trace replay."""

import argparse
import csv
import datetime as dt
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import v2_lite as runner


FIELDS = ("kind timestamp source stage base operation pattern applications phase_minutes case_repetitions "
          "calls total_ticks measured_ticks repeats warmup_runs warmup_seconds median_seconds max_seconds "
          "ticks_per_second estimated_seconds safety_factor conservative_seconds workers planned_workers cpu physical_cores logical_cpus "
          "ram_gib python numpy threads engine jit_compile_seconds seed status notes").split()


def metadata(config):
    """Read original 150 NPZ NPY headers and scalar splits, never demand bodies/confirm IDs."""
    result = {}
    for source in runner.SOURCES:
        path = ROOT / config["io"]["exploration_data"] / f"{source}.npz"
        with ZipFile(path) as archive, archive.open("demand.npy") as stream:
            version = np.lib.format.read_magic(stream)
            reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                      else np.lib.format.read_array_header_2_0 if version == (2, 0) else None)
            if reader is None:
                raise ValueError("Unsupported NPY metadata version")
            shape, _, _ = reader(stream)
        with np.load(path, allow_pickle=False) as arrays:
            fit, cal, val = [int(arrays[name]) for name in ("fit_end", "cal_end", "val_end")]
        if shape[0] != 50 or (fit, cal, val) != (4320, 7200, 11520):
            raise ValueError("Original exploration metadata differs")
        result[source] = dict(applications=shape[0], validation=val-cal, test=shape[1]-val,
                              total=shape[1], fit=fit, cal=cal, val=val)
    return result


def timed(function, ticks, repeats):
    started = time.perf_counter()
    function()  # Warm caches/imports and the actual per-method execution path.
    warm = time.perf_counter() - started
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        times.append(time.perf_counter() - started)
    return dict(measured_ticks=ticks, repeats=repeats, warmup_runs=1, warmup_seconds=warm,
                median_seconds=statistics.median(times), max_seconds=max(times),
                ticks_per_second=ticks / statistics.median(times))


def benchmark(config, ticks=4096, repeats=3):
    """Artificial high-variation proposals and demand; no real application values."""
    rng = np.random.default_rng(20261021)
    start = 7200
    case = next(case for case in runner.scenarios(config) if case["name"] == "delay600")
    grid, measurements = runner.candidate_grid(config), []
    for pattern in ("dense_uniform", "alternating_bursts"):
        y = (rng.uniform(.01, 16., start + ticks) if pattern == "dense_uniform" else
             np.resize(np.r_[np.full(30, .05), np.full(30, 16.)], start + ticks))
        y[:4320:64] = 16.  # c=2, bounded by 32; this is not a fitted dataset.
        episode = runner.prepare_episode(y, 4320, start, len(y), case, config)
        for base in (*runner.BASES, runner.COMPETITOR):
            cfg = next((cfg for cfg in grid[base] if cfg["window"] == 2880 and cfg["risk"] == .01),
                       grid[base][-1])
            if base != runner.COMPETITOR:
                measurements.append(dict(base=base, operation="prepare", pattern=pattern,
                                         **timed(lambda: runner.proposal_path(episode, base, cfg, config), ticks, repeats)))
                proposals = runner.proposal_path(episode, base, cfg, config)
            else:
                proposals = None
            measurements.append(dict(base=base, operation="raw", pattern=pattern, **timed(
                lambda: runner.evaluate_episode(episode, base, cfg, config, proposals=proposals), ticks, repeats)))
            if base in runner.BASES:
                measurements.append(dict(base=base, operation="wrapped", pattern=pattern, **timed(
                    lambda: runner.evaluate_episode(episode, base, cfg, config, .5, proposals), ticks, repeats)))
            print(f"Synthetic timing complete: {pattern}/{base}", flush=True)
    return measurements


def workload(config, sources, measurements, factor):
    """Sequential work includes cold nominal grids and shared proposal preparation."""
    rows = []
    for stage in ("explore", "confirm"):
        case_repeats = sum(5 if case["transform"] == "flash" else 1
                           for case in runner.execution_cases(config, stage))
        for source, data in sources.items():
            apps = data["applications"] if stage == "explore" else runner.CONFIRM_COUNTS[source]
            for base in (*runner.BASES, runner.COMPETITOR):
                for phase in ("validation", "test"):
                    n = data[phase]
                    # Reuse is independently audited; 60 new nominal candidates/app remain.
                    calls = apps * (30 if phase == "validation" else case_repeats)
                    if stage == "explore" and phase == "validation" and base in runner.BASES[:3]:
                        calls = 0
                    for operation in (("prepare", "raw", "wrapped") if base in runner.BASES else ("raw",)):
                        if operation == "wrapped" and phase == "validation":
                            continue
                        op_calls = calls * (5 if stage == "explore" else 1) if operation == "wrapped" else calls
                        measure = measurements[base, operation]
                        estimated = op_calls * n * measure["median_seconds"] / measure["measured_ticks"]
                        conservative = op_calls * n * measure["max_seconds"] / measure["measured_ticks"] * factor
                        rows.append(dict(kind="workload", source=source, stage=stage, base=base, operation=operation,
                                         applications=apps, phase_minutes=n, case_repetitions=case_repeats,
                                         calls=op_calls, total_ticks=op_calls*n, estimated_seconds=estimated,
                                         safety_factor=factor, conservative_seconds=conservative,
                                         status="ESTIMATE_NOT_RUNTIME",
                                         notes=phase + ("; audited three-base nominal reuse" if stage == "explore" else
                                                       "; confirm length assumed equal to same-source old trace, cohort NOT_PREPARED")))
    return rows


def analysis_timings(config):
    """Bound the nine-variant CI analysis using artificial paired summaries only."""
    import copy
    from scripts import analyze_v2_lite as analysis
    rows = []
    for stage in ("explore", "confirm"):
        counts = [50] * 3 if stage == "explore" else [runner.CONFIRM_COUNTS[s] for s in runner.SOURCES]
        data = {}
        for source, n in zip(runner.SOURCES, counts):
            values = np.zeros((n, 4, 4, 2, 2))
            values[..., 0] = .05
            values[:, :, :2, 0, 0] = .08
            values[..., 1] = 10.
            values[:, :, :2, 1, 1] = 12.
            values[:, :, 2:, 1, 1] = 9.
            data[source] = dict(ids=tuple(f"artificial-{i}" for i in range(n)), values=values,
                                paid=values[..., 1] * 1000, contract=np.ones((n, 4), dtype=bool),
                                available=np.ones((n, 4), dtype=bool), validation_ok=np.ones((n, 4), dtype=bool))
        reduced = copy.deepcopy(config)
        reduced["statistics"].update(bootstrap_repetitions=30, maximum_bootstrap_attempts=150)
        start = time.perf_counter()
        result = [analysis.bootstrap(data, reduced, variant) for variant in analysis.VARIANTS]
        elapsed = time.perf_counter() - start
        if not all(item["status"] == "OK" for item in result):
            raise AssertionError("Artificial analysis timing must have estimable intervals")
        estimate = elapsed * config["statistics"]["bootstrap_repetitions"] / 30
        rows.append(dict(kind="analysis", stage=stage, base="nine_variants", operation="paired_bootstrap",
                         applications=sum(counts), repeats=30, median_seconds=elapsed,
                         estimated_seconds=estimate, safety_factor=config["feasibility"]["runtime_safety_factor"],
                         conservative_seconds=estimate * config["feasibility"]["runtime_safety_factor"],
                         status="SYNTHETIC_EXTRAPOLATION", workers=1,
                         planned_workers=config["design"]["workers"],
                         notes="one artificial nine-variant analysis at 30 draws, scaled to 2000; not scientific CI; serial estimate"))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--safety-factor", type=float)
    parser.add_argument("--analysis-only", action="store_true", help="Append artificial CI timing to the existing workload CSV")
    args = parser.parse_args(argv)
    config = runner.load_config()
    if args.analysis_only:
        output = ROOT / config["feasibility"]["benchmark_csv"]
        with output.open(encoding="utf-8", newline="") as stream:
            existing = list(csv.DictReader(stream))
        additions = analysis_timings(config)
        with output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(row for row in existing if row["kind"] != "analysis")
            writer.writerows(dict(timestamp=dt.datetime.now(dt.timezone.utc).isoformat(), **row) for row in additions)
        for row in additions:
            print(row["stage"], "synthetic nine-variant analysis estimate", round(row["estimated_seconds"], 3), "seconds")
        return
    args.ticks = args.ticks if args.ticks is not None else config["feasibility"]["benchmark_ticks"]
    args.repeats = args.repeats if args.repeats is not None else config["feasibility"]["benchmark_repetitions"]
    args.safety_factor = args.safety_factor if args.safety_factor is not None else config["feasibility"]["runtime_safety_factor"]
    if args.ticks < 2880 or args.repeats < 3 or args.safety_factor < 2:
        parser.error("Use at least 2880 ticks, three repetitions, and safety factor >=2")
    runner.plan(config)
    sources = metadata(config)
    measured = benchmark(config, args.ticks, args.repeats)
    measurements = {}
    for item in measured:
        key = item["base"], item["operation"]
        prior = measurements.get(key, item)
        measurements[key] = dict(item, median_seconds=max(prior["median_seconds"], item["median_seconds"]),
                                 max_seconds=max(prior["max_seconds"], item["max_seconds"]))
    hardware = subprocess.check_output(["powershell", "-NoProfile", "-Command",
        "$p=Get-CimInstance Win32_Processor; $m=Get-CimInstance Win32_ComputerSystem; "
        "'{0}|{1}|{2}|{3}' -f $p.Name,$p.NumberOfCores,$p.NumberOfLogicalProcessors,($m.TotalPhysicalMemory/1GB)"],
        encoding="utf-8").strip().split("|")
    import os
    common = dict(timestamp=dt.datetime.now(dt.timezone.utc).isoformat(), workers=1,
                  planned_workers=config["design"]["workers"], cpu=hardware[0],
                  physical_cores=hardware[1], logical_cpus=os.cpu_count(), ram_gib=hardware[3],
                  python=platform.python_version(), numpy=np.__version__, engine="reference_python",
                  threads="single worker; no JIT or parallel kernel invoked", jit_compile_seconds=0, seed=20261021)
    rows = [dict(kind="metadata", source=source, applications=data["applications"], phase_minutes=data["test"],
                 status="HEADER_AND_SPLITS_ONLY", notes=f"shape=50x{data['total']}; splits=4320/7200/11520")
            for source, data in sources.items()]
    rows += [dict(kind="benchmark", status="SYNTHETIC_TIMED", **measure) for measure in measured]
    rows += workload(config, sources, measurements, args.safety_factor)
    for stage in ("explore", "confirm"):
        entries = [row for row in rows if row["kind"] == "workload" and row["stage"] == stage]
        totals = {field: sum(row[field] for row in entries) for field in
                  ("total_ticks", "estimated_seconds", "conservative_seconds")}
        rows.append(dict(kind="total", stage=stage, safety_factor=args.safety_factor, status="ESTIMATE_NOT_RUNTIME",
                         notes="serial budget, not divided by planned workers; sum(pre-processing + raw + wrapped); excludes data IO/SQLite/export; factor covers overhead", **totals))
        print(stage, {key: round(value, 3) for key, value in totals.items()}, flush=True)
    output = ROOT / config["feasibility"]["benchmark_csv"]
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(dict(common, **row) for row in rows)
    print(output, flush=True)


if __name__ == "__main__":
    main()
