"""Deterministic synthetic matrix for the passive contract monitor.

This module performs no trace, container, network or Kubernetes IO.  It is a
bounded engineering check and must not be interpreted as a service experiment.
"""

import argparse
import csv
import random
from pathlib import Path

from sysbench.control import ContractMonitor


POSITIONS = (12, 36, 60, 84, 108, 132, 156, 180, 204)
FAULTS = ("startup_bound", "runtime_cap", "demand_envelope")


def _quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _fault_latency(kind, position):
    monitor = ContractMonitor(10., clock=lambda: 0.)
    pod_id = f"synthetic-{position}"
    truth_tick = position + 8 if kind == "startup_bound" else position
    for tick in range(240):
        now = 5. * tick
        y_count = 401 if kind == "demand_envelope" and tick == position else 0
        runtime_cap = 4 if kind == "runtime_cap" and tick == position else 8
        monitor.observe_tick(tick, y_count, 1, runtime_cap=runtime_cap, when=now)
        if kind == "startup_bound" and tick == position:
            monitor.pod_requested(pod_id, when=now)
    flag = monitor.flags.get(kind)
    return None if flag is None else int(flag["tick"]) - truth_tick


def _valid_flags(seed):
    rng = random.Random(seed)
    monitor = ContractMonitor(10., clock=lambda: 0.)
    requested = rng.randrange(12, 190)
    duration_ticks = rng.randrange(1, 8)
    ready = requested + duration_ticks
    pod_id = f"valid-{seed}"
    for tick in range(240):
        now = 5. * tick
        if tick == ready:
            monitor.pod_ready(pod_id, when=now)
        monitor.observe_tick(tick, rng.randrange(401), rng.randrange(1, 9),
                             runtime_cap=8, external_request=rng.randrange(1, 9), when=now)
        if tick == requested:
            monitor.pod_requested(pod_id, when=now)
    return tuple(sorted(monitor.flags))


def matrix_rows():
    rows = []
    for kind in FAULTS:
        latencies = [_fault_latency(kind, position) for position in POSITIONS]
        detected = [value for value in latencies if value is not None]
        rows.append(dict(record="fault", fault_type=kind, trajectories=len(latencies),
                         detected=len(detected), min_delay_ticks=min(detected) if detected else "",
                         median_delay_ticks=_quantile(detected, .5) if detected else "",
                         p90_delay_ticks=_quantile(detected, .9) if detected else "",
                         max_delay_ticks=max(detected) if detected else "",
                         false_positive_trajectories="", false_positive_flags="",
                         status="DESCRIPTIVE_COMPLETE" if len(detected) == len(latencies)
                         else "DETECTION_MISSING"))
    false_flags = [_valid_flags(seed) for seed in range(20261027, 20262027)]
    false_trajectories = sum(bool(flags) for flags in false_flags)
    rows.append(dict(record="valid", fault_type="none", trajectories=len(false_flags), detected="",
                     min_delay_ticks="", median_delay_ticks="", p90_delay_ticks="",
                     max_delay_ticks="", false_positive_trajectories=false_trajectories,
                     false_positive_flags=sum(len(flags) for flags in false_flags),
                     status="PASS" if false_trajectories == 0 else "FAIL_FALSE_POSITIVE"))
    return rows


def write_rows(path):
    rows = matrix_rows()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def matrix_status(rows):
    complete_detection = all(row["status"] == "DESCRIPTIVE_COMPLETE" for row in rows if row["record"] == "fault")
    no_false_positives = all(row["status"] == "PASS" for row in rows if row["record"] == "valid")
    return "PASS" if complete_detection and no_false_positives else "FAIL"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Output summary CSV")
    args = parser.parse_args(argv)
    rows = write_rows(args.output)
    status = matrix_status(rows)
    print(f"{status}: synthetic monitor matrix; fault trajectories=27; valid trajectories=1000")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
