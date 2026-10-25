"""Independent local v3 event/request analysis; never runs containers or traffic.

Bare tick rows retain an explicit arithmetic-only path. Real runs require the
meta/tick/pod_event/complete schema. Missing evidence is rejected or labelled,
never replaced with tick-start Ready counts or online contract flags.
"""
import argparse
import bisect
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics


ARMS = {"HPA-locked", "HPA-wrapped", "HPA-conservative"}


def number(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(name + " must be finite and >= " + str(minimum))
    return value


def count(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(name + " must be a nonnegative integer")
    return value


def finite_record(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Use explicit INF for failed latency, not nonfinite JSON numbers")
    if isinstance(value, dict):
        for item in value.values():
            finite_record(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite_record(item)


def p95(values):
    return sorted(values)[math.ceil(.95 * len(values)) - 1] if values else None


def latency_value(value):
    return math.inf if value == "INF" else None if value is None else number(value, "latency")


def same_latency(recorded, rebuilt):
    recorded = latency_value(recorded)
    return recorded == rebuilt or (recorded is not None and rebuilt is not None
                                    and math.isfinite(recorded) and math.isfinite(rebuilt)
                                    and abs(recorded - rebuilt) <= 1e-9)


def analyze(path, capacity_rps=None, *, expected_ticks=240):
    path = Path(path)
    def reject_constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    rows = [json.loads(line, parse_constant=reject_constant)
            for line in path.read_text("utf-8").splitlines() if line.strip()]
    if not any("type" in row for row in rows):
        return analyze_rows(rows, capacity_rps, expected_ticks=expected_ticks)
    meta = [row for row in rows if row.get("type") == "meta"]
    requests = None
    schedule = None
    if len(meta) == 1 and meta[0].get("request_csv"):
        request_path = path.parent / meta[0]["request_csv"]
        with request_path.open(encoding="utf-8-sig", newline="") as stream:
            requests = list(csv.DictReader(stream))
    if len(meta) == 1 and meta[0].get("schedule_csv"):
        schedule = (path.parent / meta[0]["schedule_csv"]).read_bytes()
    result = analyze_run(rows, expected_ticks=expected_ticks, request_rows=requests, schedule_bytes=schedule)
    if capacity_rps is not None and capacity_rps != result["c_rps"]:
        raise ValueError("CLI capacity differs from recorded operational capacity")
    return result


def analyze_rows(rows, capacity_rps, *, expected_ticks=240):
    if number(capacity_rps, "capacity_rps") <= 0:
        raise ValueError("Use measured, frozen capacity_rps")
    count(expected_ticks, "expected_ticks")
    if not expected_ticks:
        raise ValueError("A nonempty planned run is required")
    if len(rows) != expected_ticks or [r["index"] for r in rows] != list(range(expected_ticks)):
        raise ValueError("Incomplete/nonconsecutive run: retain as failure, do not silently drop ticks")
    cumulative, largest = 0, 0.
    previous_lambda = minimum_lambda = 0.
    wrapped = rows[0]["lambda_t"] is not None
    for i, row in enumerate(rows):
        finite_record(row)
        count(row["index"], "index")
        count(row["y_count"], "y_count")
        count(row["n_start"], "n_start")
        count(row["loss"], "loss")
        count(row["loss_total"], "loss_total")
        loss = int(row["y_count"] > 5*capacity_rps*row["n_start"])
        if loss != row["loss"]:
            raise ValueError("Capacity units or tick-start Ready count disagree with recorded loss")
        cumulative += loss
        if row["loss_total"] != cumulative:
            raise ValueError("Cumulative loss differs from independent count")
        bound = .05 + (8/.5+220)/(i+1)
        for key, expected in (("ticks", i+1), ("dmax", 8), ("m", 4), ("L", 11), ("K", 220)):
            if count(row[key], key) != expected:
                raise ValueError("Independent certificate constant differs: " + key)
        number(row["lambda_safe"], "lambda_safe", minimum=-math.inf)
        number(row["bound_raw"], "bound_raw")
        number(row["bound_clamped"], "bound_clamped")
        largest = max(largest, abs(row["lambda_safe"]+8), abs(row["bound_raw"]-bound),
                      abs(row["bound_clamped"]-min(1., bound)))
        if (row["lambda_t"] is not None) != wrapped:
            raise ValueError("Wrapped/unwrapped identity changed during the run")
        if wrapped:
            number(row["lambda_t"], "lambda_t", minimum=-math.inf)
            number(row["lambda_final"], "lambda_final", minimum=-math.inf)
            number(row["bound_raw"], "bound_raw")
            if abs(row["lambda_t"]-previous_lambda) > 1e-9:
                raise ValueError("Lambda continuity failed")
            previous_lambda += .5*(.05-loss)
            minimum_lambda = min(minimum_lambda, previous_lambda)
            number(row["lambda_min"], "lambda_min", minimum=-math.inf)
            number(row["identity_residual"], "identity_residual", minimum=-math.inf)
            identity = (i+1)*.05 - previous_lambda/.5
            largest = max(largest, abs(row["loss_total"]-identity), abs(row["bound_raw"]-bound),
                          abs(row["lambda_final"]-previous_lambda), abs(row["lambda_min"]-minimum_lambda),
                          abs(row["identity_residual"]-(cumulative-identity)))
            if row["arithmetic_ok"] is not True:
                raise ValueError("Recorded arithmetic_ok contradicts independent update")
        elif any(row.get(key) is not None for key in ("lambda_final", "lambda_min", "identity_residual", "arithmetic_ok")):
            raise ValueError("Unwrapped lambda must be NOT_APPLICABLE/None")
    if largest > 1e-9:
        raise ValueError("Certificate arithmetic differs by more than 1e-9")
    paid = [r.get("paid_replica_seconds") for r in rows]
    p95 = [r.get("latency_window_p95_ms") for r in rows]
    return dict(status="ARITHMETIC_ONLY", ticks=len(rows), capacity_violation_rate=cumulative/len(rows),
                max_certificate_error=largest,
                paid_replica_seconds=sum(paid) if all(v is not None for v in paid) else "NOT_MEASURED",
                measured_p95_windows=sum(v is not None for v in p95),
                contract_flagged=any(not r["contract_valid"] for r in rows),
                engineering_matrix="NOT_ASSESSED")


def request_windows(rows, ticks, *, slo_ms=18.924, timeout_ms=189.2):
    """D1/D3 from the sender's request CSV, without using online window p95."""
    windows = [[] for _ in range(ticks)]
    written = failed = retried = 0
    previous = -1
    for i, row in enumerate(rows):
        parsed = {}
        for key in ("request_id", "offset_ns", "written_ns", "completed_ns", "http_status", "write_callbacks"):
            value = row[key]
            if isinstance(value, bool) or isinstance(value, float):
                raise ValueError("Request CSV integer required: " + key)
            try:
                parsed[key] = int(value)
            except (ValueError, TypeError) as error:
                raise ValueError("Request CSV integer required: " + key) from error
        offset, sent, done = (parsed[k] for k in ("offset_ns", "written_ns", "completed_ns"))
        if parsed["request_id"] != i or offset < previous or not 0 <= offset < ticks*5_000_000_000:
            raise ValueError("Missing/nonconsecutive request identity or invalid planned offset")
        previous = offset
        if sent < -1 or done < -1 or (sent >= 0 and sent < offset) or (done >= 0 and done < offset):
            raise ValueError("Invalid request monotonic times/sentinel")
        if sent >= 0 and done >= 0 and done < sent:
            raise ValueError("Completion precedes write")
        count(parsed["http_status"], "http_status")
        callbacks = count(parsed["write_callbacks"], "write_callbacks")
        if sent >= 0 and not callbacks:
            raise ValueError("Recorded write lacks a write callback")
        if not isinstance(row["error"], str):
            raise ValueError("Explicit request error string required")
        ok = (not row["error"] and sent >= 0 and done >= 0
              and 200 <= parsed["http_status"] < 300 and done-offset <= timeout_ms*1e6
              and done <= ticks*5_000_000_000+1_000_000_000)
        written += sent >= 0
        failed += not ok
        retried += callbacks > 1
        windows[offset//5_000_000_000].append((done-offset)/1e6 if ok else math.inf)
    quantiles = [p95(values) for values in windows]
    nonempty = sum(bool(values) for values in windows)
    violations = sum(value is not None and value > slo_ms for value in quantiles)
    return dict(status="REQUEST_CSV_RECOMPUTED", planned_requests=len(rows), written=written,
                failed=failed, missing_write=len(rows)-written, retried_requests=retried,
                window_counts=[len(values) for values in windows], window_p95=quantiles,
                empty_windows=ticks-nonempty, nonempty_windows=nonempty,
                slo_violation_windows=violations,
                slo_violation_rate=violations/nonempty if nonempty else None)


def verify_plan(raw, digest, requests, ticks):
    """Hash the actual immutable plan bytes, then align every request identity."""
    if raw is None:
        return "NOT_ASSESSED"
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Original schedule bytes differ from recorded SHA256")
    stream = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    if stream.fieldnames != ["request_id", "offset_ns"]:
        raise ValueError("Original plan must have request_id,offset_ns columns")
    plan = list(stream)
    previous = -1
    for i, row in enumerate(plan):
        try:
            identity, offset = int(row["request_id"]), int(row["offset_ns"])
        except (ValueError, TypeError) as error:
            raise ValueError("Invalid integer in original plan") from error
        if identity != i or offset < previous or not 0 <= offset < ticks*5_000_000_000:
            raise ValueError("Original plan identity/deadline is invalid")
        previous = offset
        if requests is not None and (i >= len(requests) or
                identity != int(requests[i]["request_id"]) or offset != int(requests[i]["offset_ns"])):
            raise ValueError("Request CSV differs from original plan ID/offset")
    if requests is None:
        return "PLAN_HASH_VERIFIED_REQUEST_OFFSETS_NOT_ASSESSED"
    if len(plan) != len(requests):
        raise ValueError("Request CSV omits/adds original plan identities")
    return "VERIFIED"


def event_evidence(events, ticks, meta):
    """D4/D7 from lifecycle/command events; no online monitor dependencies."""
    creates, ready, deleted, service, commands = {}, {}, {}, {}, {}
    truths = {}
    for event in events:
        at = number(event["at"], "event at")
        if at > meta["finish_mono"]:
            raise ValueError("Primitive event occurs after run finish")
        kind = event["event_type"]
        if kind in {"create", "ready", "delete", "first_service"}:
            pod = event["pod"]
            if not isinstance(pod, str) or not pod:
                raise ValueError("Immutable pod identity required")
        if kind == "create":
            if pod in creates or not isinstance(event["accepted"], bool):
                raise ValueError("Duplicate create or missing accepted state")
            creates[pod] = event
            if isinstance(event["command_tick"], bool) or not isinstance(event["command_tick"], int) or event["command_tick"] < -1:
                raise ValueError("command_tick must be -1 (warmup) or a nonnegative integer")
        elif kind in {"ready", "delete", "first_service"}:
            if pod not in creates or not creates[pod]["accepted"] or at < creates[pod]["at"]:
                raise ValueError("Pod event lacks an accepted earlier create")
            target = {"ready": ready, "delete": deleted, "first_service": service}[kind]
            if pod in target:
                raise ValueError("Duplicate immutable pod event: " + kind)
            target[pod] = event
            if kind == "first_service":
                observed_tick = count(event["tick"], "first-service tick")
                lag = observed_tick - creates[pod]["command_tick"]
                if count(event["discrete_lag"], "discrete_lag") != lag:
                    raise ValueError("Recorded discrete lag differs from command/service ticks")
        elif kind not in {"command", "runtime_cap"}:
            raise ValueError("Unknown primitive event type: " + str(kind))
        if kind == "command":
            index = count(event["tick"], "command tick")
            if index in commands or index >= len(ticks):
                raise ValueError("Duplicate/out-of-range issued command tick")
            commands[index] = event
            if count(event["issued_request"], "issued_request") > 8:
                truths["request_above_maximum"] = min(at, truths.get("request_above_maximum", at))
        if "runtime_cap" in event and count(event["runtime_cap"], "runtime_cap") < 8:
            truths["runtime_cap"] = min(at, truths.get("runtime_cap", at))
    if set(commands) != set(range(len(ticks))):
        raise ValueError("One independent issued-command event required per tick")
    accepted = {pod: event for pod, event in creates.items() if event["accepted"]}
    if not accepted or set(deleted) != set(accepted):
        raise ValueError("Every accepted create needs an explicit delete-issued boundary for billing")
    paid = ready_seconds = 0.
    lags, unresolved = [], []
    for pod, event in accepted.items():
        begin, end = event["at"], deleted[pod]["at"]
        if end < begin or end > meta["finish_mono"]:
            raise ValueError("Reversed/unclosed pod lifetime")
        paid += end-begin
        if pod in ready:
            first = ready[pod]["at"]
            if first > end:
                raise ValueError("Ready after delete")
            ready_seconds += end-first
        stop_waiting = min(ready[pod]["at"] if pod in ready else end, end)
        if stop_waiting-begin > 35:
            at = begin+35
            truths["startup_bound"] = min(at, truths.get("startup_bound", at))
        if event["command_tick"] >= 0:
            if pod in service:
                if pod not in ready or not ready[pod]["at"] <= service[pod]["at"] <= end:
                    raise ValueError("First-service event inconsistent with Ready lifetime")
                service_tick = service[pod]["tick"]
                if service_tick > len(ticks):
                    raise ValueError("First-service tick beyond final observed boundary")
                lower = ticks[service_tick]["boundary_mono"] if service_tick < len(ticks) else ticks[-1]["next_boundary_mono"]
                upper = ticks[service_tick]["tick_start_mono"] if service_tick < len(ticks) else meta["finish_mono"]
                if not lower <= service[pod]["at"] <= upper:
                    raise ValueError("First-service tick differs from actual boundary/Ready snapshot")
                lags.append(service[pod]["tick"]-event["command_tick"])
            else:
                unresolved.append(pod)
    for tick in ticks:
        at = tick["next_boundary_mono"]
        observed_ready = sum(pod in ready and ready[pod]["at"] <= tick["tick_start_mono"] < deleted[pod]["at"]
                             for pod in accepted)
        if observed_ready != tick["n_start"]:
            raise ValueError("Tick-start Ready count differs from independent pod lifecycle")
        if tick["y_count"] > 8*5*meta["c_rps"]:
            truths["demand_envelope"] = min(at, truths.get("demand_envelope", at))
        command = commands[tick["index"]]
        for key in ("issued_request", "runtime_cap"):
            if count(tick[key], key) != count(command[key], "event " + key):
                raise ValueError("Tick runtime state differs from independent command event: " + key)
    lag_status = ("NOT_APPLICABLE" if meta["arm"] != "HPA-wrapped" else
                     "NOT_OBSERVED_SUPPORTED" if any(lag > 8 for lag in lags) else
                     "NOT_ASSESSABLE" if unresolved or not lags else "OBSERVED_DISCRETE_LAG_SUPPORTED")
    applicability = ("NOT_APPLICABLE" if meta["arm"] != "HPA-wrapped" else
                     "OUTSIDE_OBSERVED_CONTRACT" if truths else
                     "OBSERVED_CHECKS_SUPPORTED_CONDITIONAL" if lag_status == "OBSERVED_DISCRETE_LAG_SUPPORTED" else lag_status)
    return dict(paid_replica_seconds=paid, ready_replica_seconds=ready_seconds,
                ready_seconds_definition="first_observed_ready_to_delete_issued",
                independent_truth=truths, certificate_applicability=applicability,
                discrete_lag_status=lag_status,
                max_discrete_lag=max(lags) if lags else None, unresolved_lag_pods=unresolved)


def monitor_evidence(truths, flags, boundaries):
    """D7 uses actual boundary-index distance; seconds are descriptive only."""
    if len(boundaries) < 2 or any(not math.isfinite(t) for t in boundaries) or any(a >= b for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("Finite increasing observed boundaries required")
    truth = min(truths.values(), default=None)
    first_flag = min((f["observed_at"] for f in flags.values()), default=None)
    outside = any(t is not None and not boundaries[0] <= t <= boundaries[-1] for t in (truth, first_flag))
    truth_index = bisect.bisect_left(boundaries, truth) if truth is not None and boundaries[0] <= truth <= boundaries[-1] else None
    flag_index = bisect.bisect_left(boundaries, first_flag) if first_flag is not None and boundaries[0] <= first_flag <= boundaries[-1] else None
    followup = truth_index is not None and truth_index+10 < len(boundaries)
    delay_seconds = first_flag-truth if truth is not None and first_flag is not None else None
    delay_ticks = flag_index-truth_index if truth_index is not None and flag_index is not None else None
    assessed = not outside and (truth is None or followup)
    detected = (delay_seconds is not None and delay_seconds >= 0 and flag_index < len(boundaries)
                and 0 <= delay_ticks <= 10) if assessed and followup else False if assessed and truth is None else None
    return dict(status="ASSESSED" if assessed else "NOT_ASSESSED_BOUNDARY_OUTSIDE" if outside else "NOT_ASSESSED_INSUFFICIENT_10_TICK_FOLLOWUP",
                first_independent_breach=truth, first_flag=first_flag,
                false_positive=first_flag is not None and truth is None,
                fault_exercised=truth is not None, detected_within_10_ticks=detected,
                missed_detection=(not detected) if assessed and followup else False if assessed and truth is None else None,
                first_truth_available_boundary=truth_index if truth_index is not None and truth_index < len(boundaries) else None,
                first_flag_available_boundary=flag_index if flag_index is not None and flag_index < len(boundaries) else None,
                detection_delay_ticks=delay_ticks, detection_delay_seconds=delay_seconds,
                complete_10_tick_followup=followup,
                premature_latch=delay_seconds is not None and delay_seconds < 0)


def verify_t1(records, meta, request_rows):
    """Transport evidence is checked separately from scientific outcome arithmetic."""
    extras=[r for r in records if r['type'] in {'connection','transport_ready'}]
    if meta.get('transport_scheme')!='T1':
        if extras: raise ValueError("Transport evidence has no matching T1 metadata")
        return
    ready=[r for r in extras if r['type']=='transport_ready']
    if len(ready)!=1 or any(ready[0].get(k)!=v for k,v in (
            ('scheme','T1'),('preconnected',256),('independent_transports',256),
            ('max_connections_per_transport',1),('assignment','request_id_mod_256'))):
        raise ValueError("T1 preconnection receipt incomplete")
    connections=[r for r in extras if r['type']=='connection']
    initial={}; known={}; attempts={i:{} for i in range(256)}
    for r in connections:
        slot=count(r['transport_id'],'transport id')
        if slot>=256: raise ValueError("T1 transport id outside range")
        attempt=count(r['dial_attempt'],'dial attempt')
        if attempt<1 or r['redial']!=(attempt>1): raise ValueError("T1 redial counter mismatch")
        when=number(r['observed_mono_ns'],'dial clock')
        if attempt in attempts[slot]: raise ValueError("Duplicate T1 dial attempt")
        attempts[slot][attempt]=when
        if r['success']:
            key=(slot,r['local_address'])
            known[key]=min(known.get(key,when),when)
            if attempt==1 and when<meta['start_mono']*1e9: initial[slot]=r['local_address']
    if set(initial)!=set(range(256)) or len(set(initial.values()))!=256 or request_rows is None:
        raise ValueError("T1 lacks all initial connections or actual request evidence")
    for events in attempts.values():
        if sorted(events)!=list(range(1,len(events)+1)) or [events[i] for i in sorted(events)]!=sorted(events.values()):
            raise ValueError("T1 dial attempts are not continuous chronological evidence")
    for i,r in enumerate(request_rows):
        slot=int(r.get('transport_id',-1))
        written=int(r['written_ns']); key=(slot,r.get('connection'))
        if slot!=i%256 or written>=0 and key not in known:
            raise ValueError("T1 request assignment/connection differs from dial evidence")
        if written>=0:
            # CSV stores first successful write and last GotConn. Retries can refer to different connections.
            deadline=meta['start_mono']*1e9+written
            if int(r['write_callbacks'])<=1:
                if known[key]>deadline: raise ValueError("T1 dial happened after request write")
            elif (not any(s==slot and t<=deadline for (s,_),t in known.items()) or
                  known[key]>meta['start_mono']*1e9+int(r['completed_ns'])):
                raise ValueError("T1 retry lacks earlier write/last-connection evidence")


def analyze_run(records, *, expected_ticks=240, request_rows=None, schedule_bytes=None):
    for record in records:
        finite_record(record)
        if record.get("type") not in {"meta", "tick", "pod_event", "complete", "connection", "transport_ready"}:
            raise ValueError("Unknown real-log record type")
    metas = [r for r in records if r["type"] == "meta"]
    completes = [r for r in records if r["type"] == "complete"]
    if len(metas) != 1 or len(completes) != 1 or records[-1]["type"] != "complete":
        raise ValueError("One meta and one final complete record are required")
    meta, complete = metas[0], completes[0]
    verify_t1(records,meta,request_rows)
    for key in ("run_id", "source", "app"):
        if not isinstance(meta[key], str) or not meta[key]:
            raise ValueError("Explicit string run/source/app identity required")
    if meta["arm"] not in ARMS:
        raise ValueError("Unknown arm")
    for key in ("repeat", "delay"):
        count(meta[key], key)
    digest = meta["schedule_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Explicit schedule SHA256 required")
    if number(meta["c_rps"], "c_rps") <= 0:
        raise ValueError("Positive measured capacity required; formal caller binds frozen value")
    for key, expected in (("slo_ms", 18.924), ("timeout_ms", 189.2)):
        if number(meta[key], key) != expected:
            raise ValueError("Recorded operational calibration differs from D3/config: " + key)
    if number(meta["finish_mono"], "finish_mono") < number(meta["start_mono"], "start_mono"):
        raise ValueError("Reversed run clock")
    planned = count(complete.get("planned_requests", complete.get("planned")), "planned_requests")
    for key in ("written", "failed", "missing_write"):
        if count(complete[key], "complete " + key) > planned:
            raise ValueError("Complete count exceeds planned requests: " + key)
    if complete["written"]+complete["missing_write"] != planned:
        raise ValueError("Complete written/missing counts do not partition plan")
    if "planned_requests" in complete and "planned" in complete and complete["planned"] != planned:
        raise ValueError("Ambiguous complete planned count aliases")
    for record in records:
        if "arm" in record and record["arm"] != meta["arm"]:
            raise ValueError("Mixed-arm run")
        if "run_id" in record and record["run_id"] != meta["run_id"]:
            raise ValueError("Mixed run identity")
        if "schedule_sha256" in record and record["schedule_sha256"] != digest:
            raise ValueError("Plan hash changed inside a run")
    ticks = [r for r in records if r["type"] == "tick"]
    arithmetic = analyze_rows(ticks, meta["c_rps"], expected_ticks=expected_ticks)
    if planned != sum(tick["y_count"] for tick in ticks):
        raise ValueError("Complete planned count differs from tick plan")
    wrapped = meta["arm"] == "HPA-wrapped"
    if any((r["lambda_t"] is not None) != wrapped for r in ticks):
        raise ValueError("Arm disagrees with lambda applicability")
    flags, intervals = {}, []
    for i, tick in enumerate(ticks):
        begin = number(tick["boundary_mono"], "boundary_mono")
        end = number(tick["next_boundary_mono"], "next_boundary_mono")
        snap = number(tick["tick_start_mono"], "tick_start_mono")
        if not begin <= snap <= end or end <= begin or end > meta["finish_mono"]:
            raise ValueError("Invalid adjacent tick boundaries/Ready snapshot")
        if i == 0 and begin < meta["start_mono"]:
            raise ValueError("First boundary precedes recorded run origin")
        if i and abs(begin-ticks[i-1]["next_boundary_mono"]) > 1e-9:
            raise ValueError("Nonadjacent tick boundaries")
        intervals.append(end-begin)
        if count(tick["planned_window_count"], "planned_window_count") != tick["y_count"]:
            raise ValueError("Controller y is not planned count")
        latency_value(tick["latency_window_p95_ms"])
        if tick.get("error") is not None:
            raise ValueError("Command/API failure: retain run as invalid and use D6 rerun ledger")
        current = tick["first_flags"]
        if not isinstance(current, dict) or any(name not in current for name in flags):
            raise ValueError("Monitor flags are not latched")
        for name, flag in current.items():
            at = number(flag["observed_at"], "flag observed_at")
            count(flag["tick"], "flag tick")
            if name in flags and flags[name] != flag:
                raise ValueError("First flag changed after latching")
            if at > meta["finish_mono"]:
                raise ValueError("Flag observed after run")
            flags[name] = flag
    events = [r for r in records if r["type"] == "pod_event"]
    evidence = event_evidence(events, ticks, meta)
    monitor = monitor_evidence(evidence["independent_truth"], flags,
                               [ticks[0]["boundary_mono"], *(tick["next_boundary_mono"] for tick in ticks)])
    latency = dict(status="NOT_ASSESSED", reason="Independent request CSV not supplied")
    if request_rows is not None:
        latency = request_windows(request_rows, expected_ticks, slo_ms=meta["slo_ms"], timeout_ms=meta["timeout_ms"])
        if latency["window_counts"] != [tick["y_count"] for tick in ticks]:
            raise ValueError("Planned request CSV counts differ from controller input")
        if any(not same_latency(tick["latency_window_p95_ms"], value)
               for tick, value in zip(ticks, latency["window_p95"])):
            raise ValueError("Online p95 differs from independent request-window reconstruction")
        for key in ("written", "failed", "missing_write"):
            if count(complete[key], "complete " + key) != latency[key]:
                raise ValueError("Sender complete totals disagree with request CSV: " + key)
        planned = complete.get("planned_requests", complete.get("planned"))
        if count(planned, "planned_requests") != latency["planned_requests"]:
            raise ValueError("Complete planned count differs from request CSV")
        latency["window_p95"] = ["INF" if v == math.inf else v for v in latency["window_p95"]]
    in_range = sum(4 <= interval <= 6 for interval in intervals)
    plan_status = verify_plan(schedule_bytes, digest, request_rows, expected_ticks)
    result = {key: meta[key] for key in ("run_id", "arm", "source", "app", "delay", "repeat", "schedule_sha256", "c_rps")}
    result.update(status="RUN_LOG_ANALYZED", ticks=expected_ticks,
                  capacity_violation_rate=arithmetic["capacity_violation_rate"],
                  max_certificate_error=arithmetic["max_certificate_error"],
                  latency=latency, monitor=monitor, **evidence,
                  loop_in_range_count=in_range, loop_interval_count=len(intervals),
                  loop_in_range_fraction=in_range/len(intervals), loop_run_pass=in_range/len(intervals) >= .95,
                  schedule_hash_content_verification=plan_status, engineering_matrix="NOT_ASSESSED")
    return result


def analyze_matrix(results, *, high_delay=30, expected_fragments=None):
    """D5/D6 entry point for one eligible attempt per slot; keep retry logs outside.

    Missing/duplicate slots cannot generate a successful matrix. Every supplied
    run contributes to pooled timing; this function never picks the best retry.
    """
    if high_delay not in (25, 30):
        raise ValueError("D9 only permits high-delay injection 30 or 25 seconds")
    slots, hashes, run_ids = {}, {}, set()
    for result in results:
        if result["status"] != "RUN_LOG_ANALYZED":
            raise ValueError("Matrix requires real-log analyses, not arithmetic-only rows")
        if result["run_id"] in run_ids:
            raise ValueError("Duplicate run identity")
        run_ids.add(result["run_id"])
        repeat = count(result["repeat"], "repeat")
        if repeat not in ({1, 2} if result["delay"] == high_delay else {1}):
            raise ValueError("Repeat identity differs from the frozen slot manifest")
        intervals = count(result["loop_interval_count"], "loop_interval_count")
        in_range = count(result["loop_in_range_count"], "loop_in_range_count")
        if intervals != 240 or in_range > intervals or result["loop_run_pass"] != (in_range/intervals >= .95):
            raise ValueError("Timing summary differs from the 240 observed intervals")
        if number(result["capacity_violation_rate"], "capacity_violation_rate") > 1:
            raise ValueError("Risk outside [0,1]")
        fragment = (result["source"], result["app"])
        key = (*fragment, result["arm"], result["delay"], result["repeat"])
        if key in slots:
            raise ValueError("Duplicate slot/retry: apply the D6 ledger before aggregation, never choose best outcomes")
        slots[key] = result
        if fragment in hashes and hashes[fragment] != result["schedule_sha256"]:
            raise ValueError("Different plans across arms/delays/repeats")
        hashes[fragment] = result["schedule_sha256"]
    timing_n = sum(r["loop_interval_count"] for r in results)
    timing_ok = sum(r["loop_in_range_count"] for r in results)
    timing = dict(pooled_in_range_fraction=timing_ok/timing_n if timing_n else None,
                  pooled_pass=bool(timing_n) and timing_ok/timing_n >= .99,
                  every_run_pass=all(r["loop_in_range_count"]/r["loop_interval_count"] >= .95 for r in results) if results else False)
    if any(r.get("schedule_hash_content_verification") != "VERIFIED" for r in results):
        return dict(status="NOT_ASSESSED", reason="Original plans/request offsets not independently verified", timing=timing)
    fragments = sorted(hashes)
    if expected_fragments is not None and set(fragments) != set(expected_fragments):
        return dict(status="NOT_ASSESSED", reason="Frozen fragment identities missing/extra", timing=timing)
    expected = {(source, app, arm, delay) for source, app in fragments for arm in ARMS for delay in (5, 15, high_delay)}
    groups = {}
    for key, result in slots.items():
        groups.setdefault(key[:-1], []).append(result)
    if len(fragments) != 6 or len(results) != 72 or set(groups) != expected or any(
            len(values) != (2 if key[-1] == high_delay else 1) for key, values in groups.items()):
        return dict(status="NOT_ASSESSED", reason="Incomplete 6-fragment/3-arm/72-slot matrix", timing=timing)
    def value(fragment, arm, delay, metric):
        return statistics.mean(number(r[metric], metric) for r in groups[(*fragment, arm, delay)])
    high_improved = nominal_ok = 0
    high_ratios, nominal_ratios = [], []
    for fragment in fragments:
        for delay, ratios in ((high_delay, high_ratios), (5, nominal_ratios)):
            base_cost = value(fragment, "HPA-locked", delay, "paid_replica_seconds")
            if not base_cost:
                raise ValueError("Zero paired cost denominator")
            ratios.append(value(fragment, "HPA-wrapped", delay, "paid_replica_seconds") / base_cost)
        high_improved += value(fragment, "HPA-wrapped", high_delay, "capacity_violation_rate") < value(fragment, "HPA-locked", high_delay, "capacity_violation_rate")
        nominal_ok += value(fragment, "HPA-wrapped", 5, "capacity_violation_rate") <= value(fragment, "HPA-locked", 5, "capacity_violation_rate") + .01
    high_cost, nominal_cost = statistics.median(high_ratios), statistics.median(nominal_ratios)
    return dict(status="COMPLETE_72_SLOT_AGGREGATE", timing=timing, fragment_count=6,
                high_delay=high_delay, high_improved_fragments=high_improved,
                high_direction_pass=high_improved >= 4, high_cost_ratio_median=high_cost,
                high_cost_pass=high_cost <= 1.25, nominal_within_1pp_fragments=nominal_ok,
                nominal_cost_ratio_median=nominal_cost, nominal_pass=nominal_ok >= 4 and nominal_cost <= 1.25,
                latency_aggregate="NOT_ASSESSED", fault_matrix="NOT_ASSESSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="?")
    parser.add_argument("--capacity-rps", type=float)
    args = parser.parse_args()
    if not args.log:
        print("NOT_RUN: future 72+6 matrix, run manifest, fault truth and SLO/error denominators await protocol freeze")
    else:
        print(json.dumps(analyze(args.log, args.capacity_rps), ensure_ascii=False, allow_nan=False, indent=2))
