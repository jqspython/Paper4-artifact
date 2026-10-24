"""Bounded Poisson capacity calibration, distinct from the exhausted shakedowns.

Only synthetic arrivals. All attempts are registered before creating Pods or load.
The final 300s run evaluates its first 180s for SLO and all 300s for D10.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import time
import tomllib

from sysbench.analyze import request_windows
from sysbench.gates import plan, independent_requests

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work/v3-sys"
LEDGER = ROOT / "results/v3-sys-capacity.csv"
DECISION = WORK / "capacity-v2-decision.json"
IMAGE = "python:3.11-slim-bookworm@sha256:0a310eeecf4e1f5a0743f9a6520c90c88d089c903ca5fd283f501e3a805f5f89"
GRID = (40, 50, 60, 70, 80)
FIELDS = ("number", "label", "purpose", "pods", "per_pod_rps", "seconds", "seed", "fresh_connections",
          "started_utc", "status", "slo_bad_windows", "slo_windows", "prefix_failed", "failed", "backend_max_over_mean",
          "missing_backend", "late_p99_ms", "sender_cpu_fraction", "slo_pass", "fidelity_pass", "note")


def ledger():
    if not LEDGER.exists(): return []
    with LEDGER.open(encoding="utf-8") as stream: return list(csv.DictReader(stream))


def write_rows(rows):
    temporary = LEDGER.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    temporary.replace(LEDGER)


def register(purpose, pods, rate, seconds, seed, fresh):
    if DECISION.exists(): raise RuntimeError("Calibration decision is terminal; no extra runs after it")
    rows = ledger()
    if len(rows) >= 12 or any(r["status"] == "RUNNING" for r in rows):
        raise RuntimeError("Calibration budget exhausted or unfinished attempt; never reset the ledger")
    n = len(rows)+1
    row = dict.fromkeys(FIELDS, "")
    row.update(number=n, label=f"cal-v2-{n:02d}", purpose=purpose, pods=pods, per_pod_rps=rate,
               seconds=seconds, seed=seed, fresh_connections=int(fresh), status="RUNNING",
               started_utc=datetime.now(timezone.utc).isoformat())
    write_rows([*rows, row])
    return row


def finish(row, values):
    rows = ledger(); stored = rows[int(row["number"])-1]
    if stored["label"] != row["label"] or stored["status"] != "RUNNING":
        raise ValueError("Only the active calibration can finish")
    stored.update(values); write_rows(rows)


def evaluate(rows, pods, seconds, *, combined=False, cpu_fraction=None):
    windows = request_windows(rows, seconds//5)
    horizon = 36 if combined else seconds//5
    if len(pods) != len(set(pods)) or not pods:
        raise ValueError("Exact nonempty backend Pod identities are required")
    seen = Counter(r.get("pod", "") for r in rows if r.get("pod", "") in pods)
    missing_backend = sum(r.get("pod", "") not in pods for r in rows)
    counts = [seen[p] for p in pods]
    ratio = max(counts)/(sum(counts)/len(pods)) if sum(counts) else math.inf
    bad = sum(v is not None and (v == "INF" or v > 18.924)
              for v in windows["window_p95"][:horizon])
    nonempty = sum(v is not None for v in windows["window_p95"][:horizon])
    limit = .05 if combined else .02
    prefix = [r for r in rows if int(r["offset_ns"]) < horizon*5_000_000_000]
    slo_requests = request_windows(prefix, horizon)
    slo_pass = (nonempty > 0 and bad/nonempty <= limit and slo_requests["failed"] == 0
                and (combined or (missing_backend == 0 and ratio <= 1.3))
                and slo_requests["retried_requests"] == 0)
    late = sorted((int(r["written_ns"])-int(r["offset_ns"]))/1e6
                  for r in rows if int(r["written_ns"]) >= 0)
    p99 = late[math.ceil(.99*len(late))-1] if late else math.inf
    fidelity = (len(late) == len(rows) and windows["retried_requests"] == 0
                and p99 <= 1.8924 and cpu_fraction is not None and cpu_fraction < .30)
    return dict(slo_bad_windows=bad, slo_windows=nonempty, prefix_failed=slo_requests["failed"],failed=windows["failed"],
                backend_max_over_mean=ratio if math.isfinite(ratio) else "INF", missing_backend=missing_backend,
                late_p99_ms=p99 if math.isfinite(p99) else "INF",
                sender_cpu_fraction=cpu_fraction, slo_pass=slo_pass, fidelity_pass=fidelity,
                backend_counts=dict(zip(pods, counts)), full_window_p95=[
                    "INF" if value is not None and math.isinf(value) else value for value in windows["window_p95"]])


def worker(row):
    """Linux real static calibration; not a formal slot or a trace replay."""
    from sysbench.control import ContractMonitor
    from sysbench.runtime import Kube, PodRuntime
    label = row["label"]; n = int(row["pods"]); rate = int(row["per_pod_rps"])
    seconds = int(row["seconds"]); fresh = bool(int(row["fresh_connections"]))
    check = ledger()[int(row["number"])-1]
    if check["status"] != "RUNNING" or any(str(check[k]) != str(row[k]) for k in
            ("label", "pods", "per_pod_rps", "seconds", "seed", "fresh_connections")):
        raise ValueError("Static calibration must be registered before real IO")
    os.sched_setaffinity(0, set(range(4, 16)))
    runtime = PodRuntime(Kube(WORK/"kube-access.json"), label, ContractMonitor(rate))
    schedule = WORK/(label+"-plan.csv")
    fingerprint, count = plan(schedule, n*rate, seconds, seed=int(row["seed"]))
    child = None; output = None; errors = None; before = time.monotonic(); primary_error = None
    try:
        for _ in range(n): runtime.create(5, -1)
        until = time.monotonic()+60
        while runtime.state() != (n, 0):
            if time.monotonic() > until: raise TimeoutError("Calibration Pods did not become Ready")
            time.sleep(.1)
        # Same ten low-rate warm requests, outside measurement, for every point.
        import urllib.request
        for _ in range(10):
            with urllib.request.urlopen("http://paper4-v3-sys-worker:30080/work", timeout=1) as resp: resp.read()
            time.sleep(1)
        cmd = [str(WORK/"bin/replay"), "-schedule", str(schedule), "-target",
               "http://paper4-v3-sys-worker:30080/work", "-out", str(WORK/(label+"-requests.csv")),
               "-duration", str(seconds), "-timeout-ms", "189.2", "-slo-ms", "18.924"]
        if fresh: cmd.append("-fresh-connections")
        output = (WORK/(label+"-events.jsonl")).open("w", encoding="utf-8")
        errors = (WORK/(label+"-sender-stderr.log")).open("w", encoding="utf-8")
        child = subprocess.Popen(cmd, stdout=output, stderr=errors,
                                 preexec_fn=lambda: os.sched_setaffinity(0, {0, 2}))
        until = time.monotonic()+seconds+35
        while child.poll() is None:
            if time.monotonic() > until: raise TimeoutError("Calibration sender watchdog")
            runtime.poll(); time.sleep(.25)
        if child.returncode: raise RuntimeError("Sender process failed; full raw evidence retained")
    except BaseException as error:
        primary_error = error; raise
    finally:
        failures = []
        try:
            if child and child.poll() is None:
                child.terminate()
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=5)
        except Exception as error: failures.append("sender: "+str(error))
        try: runtime.close()
        except Exception as error: failures.append("owned Pods: "+str(error))
        for stream in (output, errors):
            try:
                if stream: stream.close()
            except Exception as error: failures.append("stream: "+str(error))
        try:
            with (WORK/(label+"-events.jsonl")).open("a", encoding="utf-8") as out:
                out.write(json.dumps(dict(type="calibration_driver", parameters=row, created_pods=list(runtime.items),
                    lifecycle=runtime.events, started_mono=before, finished_mono=time.monotonic(), cleanup_failures=failures,
                    schedule_sha256=fingerprint, planned_requests=count,
                    sender_binary_sha256=hashlib.sha256((WORK/"bin/replay").read_bytes()).hexdigest()),
                    allow_nan=False)+"\n")
        except Exception as error: failures.append("evidence: "+str(error))
        if failures and primary_error is None: raise RuntimeError("Calibration cleanup/evidence incomplete: "+"; ".join(failures))


def point(purpose, rate, *, n=4, seconds=180, seed=20261024, fresh=False):
    row = register(purpose, n, rate, seconds, seed, fresh)
    label = row["label"]
    cmd = ["docker", "run", "--rm", "--name", "paper4-driver-"+label, "--network", "kind",
           "--label", "paper4-v3-run="+label, "--label", "paper4-v3-slot=capacity-"+label,
           "--cpuset-cpus", "0,2,4-15", "--mount", f"type=bind,source={ROOT},target=/src",
           "--workdir", "/src", "--entrypoint", "python", IMAGE, "-B", "-m", "sysbench.calibrate",
           "--worker", "--number", str(row["number"])]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=seconds+120)
        (WORK/(label+"-host.log")).write_text(result.stdout+"\nSTDERR\n"+result.stderr, encoding="utf-8")
        if result.returncode: raise RuntimeError("Calibration infrastructure failure: "+label)
        events = [json.loads(line) for line in (WORK/(label+"-events.jsonl")).read_text("utf-8").splitlines()]
        completed, = [r for r in events if r["type"] == "complete"]
        driver, = [r for r in events if r["type"] == "calibration_driver"]
        if completed["schedule_sha256"] != driver["schedule_sha256"]:
            raise ValueError("Independent plan identity mismatch")
        raw = list(csv.DictReader((WORK/(label+"-requests.csv")).open(encoding="utf-8")))
        schedule = list(csv.DictReader((WORK/(label+"-plan.csv")).open(encoding="ascii")))
        if len(raw) != len(schedule) or any((r["request_id"], r["offset_ns"]) !=
                (p["request_id"], p["offset_ns"]) for r, p in zip(raw, schedule)):
            raise ValueError("Independent request/plan alignment failed")
        cpu_fraction = completed["sender_cpu_seconds"]/(seconds*2)
        if abs(cpu_fraction-completed["sender_bound_cpu_fraction"]) > 1e-12:
            raise ValueError("Sender CPU normalization differs from bound two cores")
        score = evaluate(raw, driver["created_pods"], seconds, combined=purpose == "eight_pod_fidelity",
                         cpu_fraction=cpu_fraction)
        values = {k: v for k, v in score.items() if k in FIELDS}
        values.update(status="MEASURED", note="exact_D3_nearest_rank; failed_points_retained")
    except BaseException as error:
        from sysbench.execute import cleanup_failed
        stream = io.StringIO(); cleanup_error = ""
        try:
            config = tomllib.loads((ROOT/"configs/v3-sys.toml").read_text("utf-8"))
            cleanup_failed(config,dict(run_id=label,slot_id="capacity-"+label),stream,root=ROOT)
        except Exception as failure: cleanup_error = "; cleanup: "+str(failure)
        try:
            finish(row,dict(status="INFRASTRUCTURE_INVALID",note=type(error).__name__+": "+str(error)+cleanup_error))
        finally:
            try: (WORK/(label+"-cleanup.log")).write_text(stream.getvalue(),encoding="utf-8")
            except OSError: pass  # Cleanup already ran even when the evidence sink fails.
        raise
    finish(row, values)  # Exactly one terminal registration; console output cannot reclassify it.
    brief = {k:v for k,v in score.items() if k != "full_window_p95"}
    print(json.dumps(dict(number=row["number"], label=label, purpose=purpose, per_pod_rps=rate,
                          pods=n, fresh_connections=fresh, **brief), allow_nan=False), flush=True)
    return score


def run_all():
    if ledger(): raise RuntimeError("Calibration has already started; do not reset or automatically rerun it")
    grid = {}; fresh = False
    for rate in GRID:
        score = point("grid", rate, fresh=fresh)
        grid[rate] = score
        if rate == GRID[0] and float(score["backend_max_over_mean"]) > 1.3:
            fresh = True
            print("BACKEND_IMBALANCE: keep original point; all subsequent points/arms use fresh connections", flush=True)
    passing = [r for r in GRID if grid[r]["slo_pass"]]
    decision = dict(status="FAILED_NO_GRID_CAPACITY", selected_capacity=None, fresh_connections=fresh)
    if passing:
        best = max(passing); lower = best-10
        # At the lower boundary 30 is only an adjacent reference, never a selectable grid point.
        groups = {best: [], lower: []}
        for rate in (best, lower):
            for i in range(3): groups[rate].append(point("repeat", rate, seed=2026102401+i, fresh=fresh))
        valid = [r for r in (best, lower) if r in GRID and grid[r]["slo_pass"]
                 and all(v["slo_pass"] for v in groups[r])]
        decision.update(status="FAILED_REPEATED_CAPACITY", candidates=[best, lower])
        if valid:
            selected = max(valid)
            final = point("eight_pod_fidelity", selected, n=8, seconds=300, seed=2026102499, fresh=fresh)
            decision.update(status="PASS" if final["slo_pass"] and final["fidelity_pass"] else "FAILED_FINAL_NO_BUDGET_FOR_DOWNSTEP",
                            selected_capacity=selected if final["slo_pass"] and final["fidelity_pass"] else None,
                            last_tested_capacity=selected)
    decision.update(short_runs_used=len(ledger()), short_run_limit=12, formal_runs_started=False,
                    reason="formal_Poisson_arrivals_recalibration;v1_capacity_was_constant_rate")
    destination = DECISION
    if destination.exists(): raise FileExistsError("Never overwrite a completed calibration decision")
    destination.write_text(json.dumps(decision, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print("CALIBRATION_DECISION "+json.dumps(decision), flush=True)


def finalize_stopped():
    """Close a failed five-point screen after an interrupted static run; no IO to the cluster."""
    rows=ledger()
    if DECISION.exists(): raise FileExistsError("Keep the original terminal decision")
    if len(rows)!=5 or [int(r["per_pod_rps"]) for r in rows]!=list(GRID) or any(
            r["purpose"]!="grid" or r["status"] not in {"MEASURED","INFRASTRUCTURE_INVALID"} for r in rows):
        raise ValueError("Only the retained, finished five-point initial screen can use this path")
    if any(r["slo_pass"]=="True" for r in rows):
        raise ValueError("A qualified grid point needs the prescribed confirmation workflow")
    invalid=sum(r["status"]=="INFRASTRUCTURE_INVALID" for r in rows)
    minimum=len(rows)+(1 if invalid else 0)+6+1
    if invalid==0 or minimum<=12:
        raise ValueError("This path requires retained interruption and insufficient completion budget")
    decision=dict(status="FAILED_NO_QUALIFIED_CAPACITY_WITHIN_BUDGET",selected_capacity=None,
        fresh_connections=rows[-1]["fresh_connections"]=="1",short_runs_used=len(rows),short_run_limit=12,
        measured_initial_points=len(rows)-invalid,invalid_initial_points=invalid,
        minimum_runs_if_retry_then_confirm=minimum,formal_runs_started=False,
        physical_impossibility_proven=False,
        reason="No qualified measured grid point; Ready-regression abort retained; completing a retry plus confirmation exceeds 12",
        unrun_stages=["six_candidate_repeats","eight_pod_SLO_prefix","new_maximum_D10"],next_stage="FALLBACK_A")
    DECISION.write_text(json.dumps(decision,indent=2)+"\n",encoding="utf-8")
    print("CALIBRATION_DECISION "+json.dumps(decision),flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", action="store_true"); p.add_argument("--number", type=int)
    p.add_argument("--run-all", action="store_true")
    p.add_argument("--finalize-stopped",action="store_true")
    args = p.parse_args()
    if args.worker:
        if args.number is None: p.error("Worker needs a registered number")
        worker(ledger()[args.number-1])
    elif args.run_all: run_all()
    elif args.finalize_stopped: finalize_stopped()
    else: p.error("Choose --run-all or a registered Linux worker")
