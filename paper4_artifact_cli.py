"""Installable, trace-free CLIs for the Paper4 replay, wrapper and monitor."""

import argparse
import csv
import json
import math
from pathlib import Path

from sysbench.control import ContractMonitor, TickController


def _records(path=None):
    if path is None:
        return [dict(y_count=value, n_start=ready, base_request=base, runtime_cap=8)
                for value, ready, base in ((40, 1, 2), (80, 1, 3), (120, 3, 4), (80, 5, 3),
                                           (40, 5, 2), (20, 5, 1), (20, 5, 1), (20, 5, 1),
                                           (20, 5, 1), (20, 5, 1), (40, 1, 2), (40, 1, 2))]
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"y_count", "n_start", "base_request"}
    if not rows or not required <= set(rows[0]):
        raise ValueError("Replay CSV requires y_count,n_start,base_request")
    result = []
    for row in rows:
        item = dict(y_count=int(row["y_count"]), n_start=int(row["n_start"]),
                    base_request=float(row["base_request"]), runtime_cap=int(row.get("runtime_cap") or 8))
        if row.get("external_request") not in (None, ""):
            item["external_request"] = int(row["external_request"])
        result.append(item)
    return result


def replay_records(records, *, capacity_rps=10., wrapped=True):
    monitor = ContractMonitor(capacity_rps, clock=lambda: 0.)
    controller = TickController(monitor, wrapped=wrapped)
    commands, ticks = [], []
    for index, row in enumerate(records):
        emitted = []
        tick = controller.finish_tick(index, row["y_count"], row["n_start"], row["base_request"],
                                      ready_now=row["n_start"], inflight=0, emit=emitted.append,
                                      runtime_cap=row.get("runtime_cap", 8),
                                      external_request=row.get("external_request"), when=float(index))
        commands.extend(emitted); ticks.append(tick)
    snapshot = monitor.snapshot(lambda_final=controller.value if wrapped else None,
                                lambda_min=controller.minimum_value if wrapped else None)
    return dict(mode="wrapped" if wrapped else "base", commands=commands, ticks=len(ticks), snapshot=snapshot)


def replay_main(argv=None):
    parser = argparse.ArgumentParser(description="Replay tick records through the controller without a cluster")
    parser.add_argument("--input", help="CSV with y_count,n_start,base_request and optional runtime_cap/external_request")
    parser.add_argument("--capacity-rps", type=float, default=10.)
    parser.add_argument("--base", action="store_true", help="Disable the feedback wrapper")
    args = parser.parse_args(argv)
    print(json.dumps(replay_records(_records(args.input), capacity_rps=args.capacity_rps,
                                    wrapped=not args.base), sort_keys=True))


def wrapper_main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate one feedback-wrapper decision")
    parser.add_argument("--y-count", type=int, required=True)
    parser.add_argument("--n-start", type=int, required=True)
    parser.add_argument("--base-request", type=float, required=True)
    parser.add_argument("--capacity-rps", type=float, default=10.)
    parser.add_argument("--lambda-value", type=float, default=0.)
    args = parser.parse_args(argv)
    monitor = ContractMonitor(args.capacity_rps, clock=lambda: 0.)
    controller = TickController(monitor, wrapped=True)
    if not math.isfinite(args.lambda_value):
        raise ValueError("lambda-value must be finite")
    controller.value = min(0., args.lambda_value) if args.lambda_value < 0 else args.lambda_value
    controller.minimum_value = min(0., controller.value)
    emitted = []
    result = controller.finish_tick(0, args.y_count, args.n_start, args.base_request,
                                    ready_now=args.n_start, inflight=0, emit=emitted.append, when=0.)
    print(json.dumps(dict(result=result, emitted=emitted), sort_keys=True))


def monitor_main(argv=None):
    parser = argparse.ArgumentParser(description="Recompute contract flags and certificate from tick records")
    parser.add_argument("--input", help="CSV accepted by paper4-replay; defaults to a trace-free synthetic fixture")
    parser.add_argument("--capacity-rps", type=float, default=10.)
    args = parser.parse_args(argv)
    monitor = ContractMonitor(args.capacity_rps, clock=lambda: 0.)
    for index, row in enumerate(_records(args.input)):
        monitor.observe_tick(index, row["y_count"], row["n_start"], runtime_cap=row.get("runtime_cap", 8),
                             external_request=row.get("external_request"), when=float(index))
    print(json.dumps(monitor.snapshot(), sort_keys=True))


def synthetic_main(argv=None):
    parser = argparse.ArgumentParser(description="Run the bounded, trace-free artifact check")
    parser.parse_args(argv)
    result = replay_records(_records(), capacity_rps=10., wrapped=True)
    snapshot = result["snapshot"]
    if (not snapshot["arithmetic_ok"] or not snapshot["contract_valid"]
            or snapshot["certificate_status"] != "CONDITIONAL_ON_EXECUTION_CONTRACT"):
        raise RuntimeError("Synthetic controller/monitor check failed")
    print(json.dumps(dict(status="PASS_SYNTHETIC_ONLY", ticks=result["ticks"],
                          loss_total=snapshot["loss_total"], identity_residual=snapshot["identity_residual"]),
                     sort_keys=True))


if __name__ == "__main__":
    synthetic_main()
