"""Bounded kind check of two execution-contract assumptions; no HTTP load.

The check is independent of the permanently cancelled 78-slot matrix.  It may
run only from its frozen design tag, no earlier than 2026-10-28, and records at
most two whole-matrix invalid attempts in a new ledger.
"""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

from sysbench.control import ContractMonitor, TickController
from sysbench.runtime import Kube, PodRuntime


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work/v3-sys"
IMAGE = "python:3.11-slim-bookworm@sha256:0a310eeecf4e1f5a0743f9a6520c90c88d089c903ca5fd283f501e3a805f5f89"
TAG = "v1.1-engineering-additions-design-v1"
LEDGER = ROOT / "results/engineering-additions/e-a-kind-attempts.csv"
PODS = ROOT / "results/engineering-additions/e-a-kind-pods.csv"
SUMMARY = ROOT / "results/engineering-additions/e-a-kind-summary.csv"
DOWNSCALES = ROOT / "results/engineering-additions/e-a-kind-downscale.csv"
PROGRESS = ROOT / "results/engineering-additions/e-a-kind-progress.csv"
DELAYS = (120, 300)
REPETITIONS = 3
SERVICE_IMAGE = "paper4-v3-http:poisson-v2"
NODES = ("paper4-v3-sys-control-plane", "paper4-v3-sys-worker", "paper4-v3-sys-worker2")
LEDGER_FIELDS = ("attempt", "design_commit", "receipt_sha256", "started_local", "finished_local", "status", "note")
SHANGHAI = timezone(timedelta(hours=8), "Asia/Shanghai")
PROTECTED = ("docs/engineering-additions-protocol.md", "sysbench/contract_assumption.py",
             "sysbench/control.py", "sysbench/runtime.py")


def design_rows():
    return [dict(delay_seconds=delay, model_d=1 + math.ceil(delay / 60), repetition=repeat,
                 initial_pods=1, final_pods=5, new_pods=4, pending_downscale_checks=2)
            for delay in DELAYS for repeat in range(1, REPETITIONS + 1)]


def observed_steps(seconds):
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Startup duration must be finite and nonnegative")
    return math.ceil(seconds / 60)


def _write(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or tuple(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _read_ledger():
    if not LEDGER.exists():
        return []
    with LEDGER.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _receipt_path(attempt):
    return WORK / f"e-a-source-receipt-{attempt}.csv"


def _time_gate(now=None):
    current = now or datetime.now(SHANGHAI)
    if current < datetime(2026, 10, 28, tzinfo=SHANGHAI):
        raise RuntimeError("E-A is frozen for execution no earlier than 2026-10-28 Asia/Shanghai")
    if current >= datetime(2026, 10, 29, 12, tzinfo=SHANGHAI):
        raise RuntimeError("E-A deadline passed; omit the check from the manuscript")


def _frozen_source():
    head = _git("rev-parse", "HEAD")
    frozen = _git("rev-list", "-n", "1", TAG)
    if head != frozen:
        raise RuntimeError(f"E-A source differs from {TAG}; do not run an amended design")
    for name in PROTECTED:
        current = (ROOT / name).read_bytes()
        tagged = subprocess.check_output(["git", "show", f"{TAG}:{name}"], cwd=ROOT)
        if hashlib.sha256(current).digest() != hashlib.sha256(tagged).digest():
            raise RuntimeError("Protected E-A source differs from the design tag: " + name)


def _write_receipt(attempt):
    commit = _git("rev-list", "-n", "1", TAG)
    rows = [dict(attempt=attempt, tag=TAG, commit=commit, path=name,
                 sha256=hashlib.sha256((ROOT / name).read_bytes()).hexdigest()) for name in PROTECTED]
    _write(_receipt_path(attempt), rows)
    return commit, hashlib.sha256(_receipt_path(attempt).read_bytes()).hexdigest()


def _verify_receipt(attempt):
    rows = _read(_receipt_path(attempt))
    ledger = _read_ledger()
    bound = ledger[-1] if ledger else {}
    if len(rows) != len(PROTECTED) or {row["path"] for row in rows} != set(PROTECTED):
        raise RuntimeError("Missing or incomplete host-bound E-A source receipt")
    if (hashlib.sha256(_receipt_path(attempt).read_bytes()).hexdigest() != bound.get("receipt_sha256")
            or any(row["commit"] != bound.get("design_commit") for row in rows)):
        raise RuntimeError("E-A source receipt is not bound to the active ledger row")
    for row in rows:
        if (row["attempt"] != str(attempt) or row["tag"] != TAG
                or hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() != row["sha256"]):
            raise RuntimeError("E-A source receipt mismatch: " + row["path"])


def _authorized(mode):
    attempt = os.environ.get("PAPER4_EA_ATTEMPT", "")
    if not attempt.isdigit() or os.environ.get("PAPER4_EA_MODE") != mode:
        raise RuntimeError("Internal E-A entry requires the registered host launcher")
    rows = _read_ledger()
    if not rows or rows[-1]["attempt"] != attempt or rows[-1]["status"] != "STARTED":
        raise RuntimeError("Internal E-A entry is not bound to the active STARTED attempt")
    _verify_receipt(int(attempt))
    return int(attempt)


def _preloaded():
    for node in NODES:
        running = subprocess.check_output(["docker", "inspect", "-f", "{{.State.Running}}", node], text=True).strip()
        if running != "true":
            raise RuntimeError("kind node is not running: " + node)
        images = subprocess.check_output(["docker", "exec", node, "crictl", "images", "-q", SERVICE_IMAGE], text=True).strip()
        if not images:
            raise RuntimeError("Service image is not preloaded on " + node)


def _wait_state(runtime, expected, timeout):
    deadline = time.monotonic() + timeout
    while True:
        state = runtime.state()
        if state == expected:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Expected Pod state {expected}; observed {state}")
        time.sleep(.5)


def _wait_target(runtime, expected, timeout):
    try:
        _wait_state(runtime, expected, timeout)
        return True
    except TimeoutError:
        return False


def _wait_empty(api, owner, timeout=30):
    deadline = time.monotonic() + timeout
    while True:
        items = api.call("GET", "pods?labelSelector=paper4-run%3D" + owner)["items"]
        if not items:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Owned Pods did not disappear after cleanup")
        time.sleep(.5)


def _pod_rows(attempt, delay, repeat, runtime, initial, recorded):
    rows = []
    allowance = 1 + math.ceil(delay / 60)
    for name, saved in runtime.items.items():
        if name == initial or name in recorded:
            continue
        ready = saved["ready"]
        observed = time.monotonic()
        elapsed = observed - saved["created"]
        seconds = "" if ready is None else ready - saved["created"]
        steps = "" if ready is None else observed_steps(seconds)
        censored = ready is None and elapsed < allowance * 60
        row = dict(attempt=attempt, delay_seconds=delay, model_d=allowance, repetition=repeat,
                   pod=name, request_mono_seconds=f"{saved['created']:.9f}",
                   ready_mono_seconds="" if ready is None else f"{ready:.9f}",
                   startup_seconds="" if ready is None else f"{seconds:.9f}", observed_steps=steps,
                   elapsed_lower_bound_seconds="" if ready is not None else f"{elapsed:.9f}",
                   within_model_d="" if censored else 0 if ready is None else int(steps <= allowance),
                   outcome="READY" if ready is not None else
                   "CENSORED_BEFORE_MODEL_D" if censored else "NOT_READY_BY_MODEL_D")
        _append(PODS, row); rows.append(row); recorded.add(name)
    return rows


def _repetition(attempt, delay, repeat):
    owner = f"e-a-a{attempt}-d{delay}-r{repeat}"
    monitor = ContractMonitor(10., startup_bound_seconds=360)
    api = Kube(WORK / "kube-access.json")
    runtime = PodRuntime(api, owner, monitor)
    controller = TickController(monitor, wrapped=False)
    initial = None; rows = []; recorded = set(); blocked = attempted = 0
    try:
        initial = runtime.create(0, -1)
        _wait_state(runtime, (1, 0), 60)
        _append(PROGRESS, dict(attempt=attempt, delay_seconds=delay, repetition=repeat,
                              event="TIMING_STARTED", owner=owner, note="before_first_accepted_addition"))
        for phase, (up_tick, block_tick, base, expected) in enumerate(((0, 1, 3., 3), (2, 3, 5., 5)), 1):
            ready, inflight = runtime.state()
            up = controller.finish_tick(up_tick, 0, ready, base, ready_now=ready, inflight=inflight,
                                        emit=lambda target, t=up_tick: runtime.issue(target, delay, t))
            if up["error"]:
                raise RuntimeError("Accepted-addition path failed after timing began: " + up["error"])
            ready, inflight = runtime.state()
            down = controller.finish_tick(block_tick, 0, ready, 1., ready_now=ready, inflight=inflight,
                                          emit=lambda target, t=block_tick: runtime.issue(target, delay, t))
            projected = ready + inflight
            is_blocked = bool(inflight and down["request"] < projected and down["command"] == projected
                              and not down["error"])
            attempted += 1; blocked += int(is_blocked)
            _append(DOWNSCALES, dict(attempt=attempt, delay_seconds=delay, repetition=repeat, phase=phase,
                                    ready=ready, pending=inflight, requested=down["request"],
                                    emitted_target=down["command"], issued=int(down["issued"]),
                                    blocked=int(is_blocked), error=down["error"] or ""))
            reached = _wait_target(runtime, (expected, 0), delay + 60)
            rows.extend(_pod_rows(attempt, delay, repeat, runtime, initial, recorded))
            if not reached:
                return rows, blocked, attempted
        return rows, blocked, attempted
    finally:
        failures = []
        try:
            rows.extend(_pod_rows(attempt, delay, repeat, runtime, initial, recorded))
        except Exception as error:
            failures.append("incremental evidence: " + str(error))
        try:
            runtime.close()
        except Exception as error:
            failures.append("owned Pod cleanup: " + str(error))
        try:
            _wait_empty(api, owner)
        except Exception as error:
            failures.append("cleanup observation: " + str(error))
        if failures:
            raise RuntimeError("; ".join(failures))


def worker():
    attempt = _authorized("worker")
    pod_rows = []
    blocked = attempted = 0
    try:
        for design in design_rows():
            rows, good, total = _repetition(attempt, design["delay_seconds"], design["repetition"])
            pod_rows.extend(rows); blocked += good; attempted += total
        persisted = [row for row in _read(PODS) if int(row["attempt"]) == attempt]
        persisted_downscales = [row for row in _read(DOWNSCALES) if int(row["attempt"]) == attempt]
        status = assumption_status(persisted, persisted_downscales)
        note = ""
    except BaseException as error:
        if not any(record.get("attempt") == str(attempt) for record in _read(PROGRESS)):
            raise
        status = "INCOMPLETE_AFTER_TIMING"
        note = type(error).__name__ + ": " + str(error)
    completed = datetime.now(SHANGHAI)
    if completed >= datetime(2026, 10, 29, 12, tzinfo=SHANGHAI):
        status = "INCOMPLETE_AFTER_DEADLINE"
        note = (note + "; " if note else "") + "Complete evidence did not exist before the frozen deadline"
    summaries = []
    persisted = [row for row in _read(PODS) if int(row["attempt"]) == attempt]
    for delay in DELAYS:
        rows = [row for row in persisted if int(row["delay_seconds"]) == delay]
        values = sorted(float(row["startup_seconds"]) for row in rows if row["startup_seconds"] != "")
        summaries.append(dict(attempt=attempt, delay_seconds=delay, model_d=1 + math.ceil(delay / 60),
            accepted_new_pods=len(rows), ready_pods=len(values),
            startup_min_seconds=f"{min(values):.9f}" if values else "",
            startup_median_seconds=f"{_median(values):.9f}" if values else "",
            startup_max_seconds=f"{max(values):.9f}" if values else "",
            pods_within_model_d=sum(row["within_model_d"] == "1" for row in rows),
            pods_exceeding_model_d=sum(row["within_model_d"] == "0" for row in rows),
            pods_censored=sum(row["within_model_d"] == "" for row in rows),
            downscale_attempts=sum(1 for row in _read(DOWNSCALES) if int(row["attempt"]) == attempt and int(row["delay_seconds"]) == delay),
            downscale_blocked=sum(int(row["blocked"]) for row in _read(DOWNSCALES)
                                  if int(row["attempt"]) == attempt and int(row["delay_seconds"]) == delay),
            completed_local=completed.isoformat(), status=status, note=note))
    _write(SUMMARY, summaries)
    print(status)


def _read(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _median(values):
    center = len(values) // 2
    return values[center] if len(values) % 2 else (values[center - 1] + values[center]) / 2


def assumption_status(pods, downscales):
    complete = (len(pods) == 24 and len(downscales) == 12
                and all(row["within_model_d"] in ("0", "1") for row in pods))
    if not complete:
        return "INCOMPLETE_AFTER_TIMING"
    held = (all(row["within_model_d"] == "1" for row in pods)
            and all(row["blocked"] == "1" for row in downscales))
    return "ASSUMPTIONS_HELD_IN_THIS_CHECK" if held else "ASSUMPTIONS_NOT_ALWAYS_HELD"


def cleanup():
    attempt = _authorized("cleanup")
    api = Kube(WORK / "kube-access.json")
    prefix = f"e-a-a{attempt}-"
    items = api.call("GET", "pods?labelSelector=app%3Dpaper4-cpu-http")["items"]
    for item in items:
        name = item["metadata"]["name"]
        if name.startswith(prefix):
            api.call("DELETE", "pods/" + name, {"gracePeriodSeconds": 0})
    deadline = time.monotonic() + 30
    while True:
        left = [item["metadata"]["name"] for item in api.call(
            "GET", "pods?labelSelector=app%3Dpaper4-cpu-http")["items"]
            if item["metadata"]["name"].startswith(prefix)]
        if not left:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Exact owned Pods remain after cleanup: " + ",".join(left))
        time.sleep(.5)


def _remove_owned_container(container, attempt):
    inspected = subprocess.run(["docker", "inspect", "-f", "{{json .Config.Labels}}", container],
                               capture_output=True, text=True, timeout=15)
    if inspected.returncode:
        return ""
    labels = json.loads(inspected.stdout)
    expected = {"paper4-e-a-attempt": str(attempt), "paper4-e-a-owner": "contract-assumption"}
    if any(labels.get(key) != value for key, value in expected.items()):
        return "container_name_collision_not_owned"
    removed = subprocess.run(["docker", "rm", "-f", container], capture_output=True, text=True, timeout=30)
    return "" if removed.returncode == 0 else removed.stderr.strip() or removed.stdout.strip()


def run():
    _time_gate(); _frozen_source()
    attempts = _read_ledger()
    if any(row["status"] != "INVALID" for row in attempts):
        raise RuntimeError("E-A has a terminal timed or complete attempt; no retry is authorized")
    if len(attempts) >= 2:
        raise RuntimeError("E-A two-invalid-attempt budget is exhausted")
    now = datetime.now(SHANGHAI)
    row = dict(attempt=len(attempts) + 1, design_commit="", receipt_sha256="",
               started_local=now.isoformat(), finished_local="", status="STARTED", note="")
    attempts.append(row); _write(LEDGER, attempts, LEDGER_FIELDS)
    row["design_commit"], row["receipt_sha256"] = _write_receipt(row["attempt"])
    _write(LEDGER, attempts, LEDGER_FIELDS)
    container = f"paper4-e-a-attempt-{row['attempt']}"
    try:
        _preloaded()
        environment = ["-e", f"PAPER4_EA_ATTEMPT={row['attempt']}", "-e", "PAPER4_EA_MODE=worker"]
        process = subprocess.run(["docker", "run", "--name", container, "--network", "kind",
            "--label", f"paper4-e-a-attempt={row['attempt']}", "--label", "paper4-e-a-owner=contract-assumption", *environment,
            "--mount", f"type=bind,source={ROOT},target=/src", "--workdir", "/src",
            "--entrypoint", "python", IMAGE, "-B", "-m", "sysbench.contract_assumption", "--worker"],
            capture_output=True, text=True, encoding="utf-8", timeout=2700)
        if process.returncode:
            raise RuntimeError("Worker failed: " + (process.stderr.strip() or process.stdout.strip()))
        terminal = _read(SUMMARY)[0]["status"]
        row.update(status="COMPLETE" if terminal.startswith("ASSUMPTIONS_") else terminal,
                   note=process.stdout.strip())
    except BaseException as error:
        timed = any(record.get("attempt") == str(row["attempt"]) for record in _read(PROGRESS))
        row.update(status="INCOMPLETE_AFTER_TIMING" if timed else "INVALID",
                   note=type(error).__name__ + ": " + str(error))
        raise
    finally:
        cleanup_note = ""
        try:
            cleanup_note = _remove_owned_container(container, row["attempt"])
        except BaseException as error:
            cleanup_note = "container cleanup: " + type(error).__name__ + ": " + str(error)
        try:
            cleanup_process = subprocess.run(["docker", "run", "--rm", "--network", "kind",
                "-e", f"PAPER4_EA_ATTEMPT={row['attempt']}", "-e", "PAPER4_EA_MODE=cleanup",
                "--mount", f"type=bind,source={ROOT},target=/src", "--workdir", "/src",
                "--entrypoint", "python", IMAGE, "-B", "-m", "sysbench.contract_assumption", "--cleanup"],
                capture_output=True, text=True, encoding="utf-8", timeout=60)
            if cleanup_process.returncode:
                detail = cleanup_process.stderr.strip() or cleanup_process.stdout.strip()
                cleanup_note += ("; " if cleanup_note else "") + "Pod cleanup: " + detail
        except BaseException as error:
            cleanup_note += (("; " if cleanup_note else "") + "Pod cleanup: "
                             + type(error).__name__ + ": " + str(error))
        if cleanup_note:
            row["note"] += "; cleanup_unverified: " + cleanup_note
            row["status"] = ("INVALID_CLEANUP_UNVERIFIED" if row["status"] == "INVALID"
                             else "COMPLETE_CLEANUP_UNVERIFIED" if row["status"] == "COMPLETE"
                             else row["status"])
        row["finished_local"] = datetime.now(SHANGHAI).isoformat()
        _write(LEDGER, attempts, LEDGER_FIELDS)


def plan():
    print("delay_seconds,model_d,repetition,new_pods,pending_downscale_checks")
    for row in design_rows():
        print(f"{row['delay_seconds']},{row['model_d']},{row['repetition']},{row['new_pods']},{row['pending_downscale_checks']}")
    print("TOTAL,new_pods=24,attempt_budget=2,no_application_requests=true")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    if sum((args.plan, args.run, args.worker, args.cleanup)) != 1:
        parser.error("choose exactly one of --plan, --run, --worker or --cleanup")
    plan() if args.plan else run() if args.run else worker() if args.worker else cleanup()
