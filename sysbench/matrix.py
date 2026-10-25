"""Fixed ID-only v3 slots and append-only D6 attempts; no trace or launcher IO.

One eligible attempt is final. Pooled timing below 99% is an overall failure,
never permission to retry individually valid slots or select cheaper outcomes.
"""
import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import subprocess
import tomllib


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("HPA-locked", "HPA-wrapped", "HPA-conservative")
MANIFEST = "results/v3-sys-manifest.csv"
LEDGER = "results/v3-sys-attempts.csv"
FIELDS = ("order", "slot_id", "kind", "source", "app", "arm", "delay", "repeat", "fault",
          "arrival_fault", "seed", "plan_key", "ticks", "fault_start_tick", "fault_end_tick")
EVENT_FIELDS = (*FIELDS, "seq", "at_utc", "attempt", "event", "run_id", "freeze_tag", "freeze_commit",
                "infrastructure_ok", "in_range_ticks", "total_ticks", "reason", "log_path")
CRITICAL = ("sysbench", "configs/v3-sys.toml", "docs/v3-sys-protocol.md", "results/v3-sys-fragment-ids.csv")


def owned(root, path):
    root = Path(root).resolve()
    result = (root / path).resolve()
    if not result.is_relative_to(root):
        raise ValueError("Keep v3 artifacts inside the project root")
    return result


def canonical_ledger(root, path):
    result = owned(root, path)
    if result != owned(root, LEDGER):
        raise ValueError("Formal attempts use one canonical ledger; a new path cannot reset reruns")
    return result


def read_csv(path, fields=None):
    path = Path(path)
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if fields is not None and reader.fieldnames != list(fields):
            raise ValueError("CSV schema differs: " + str(path))
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Partial/extra CSV fields; retain the damaged evidence")
    return rows


def build_manifest(config, identities):
    """Only identities/parameters. Hash and Poisson seed match fragments.py."""
    if config["selection"]["seed"] != 20261022:
        raise ValueError("Identity/Poisson seed namespace differs from the fixed rule")
    if tuple(config["matrix"]["arms"]) != ARMS or config["matrix"]["repeats"] != [1, 1, 2]:
        raise ValueError("Frozen three-arm/repeat design changed")
    delays = config["matrix"]["injection_seconds"]
    if delays[:2] != [5, 15] or len(delays) != 3 or delays[2] not in (25, 30):
        raise ValueError("Only the declared D9 30-to-25 branch is allowed")
    if config["controller"]["ticks"] != 240 or config["matrix"]["formal_runs"] != 72 or config["matrix"]["fault_runs"] != 6:
        raise ValueError("The fixed 240-tick/72+6 design changed")
    if (config["faults"]["start_tick"], config["faults"]["end_tick_exclusive"]) != (60, 120):
        raise ValueError("D8 fault window changed")
    if len({(r["source"], r["app"]) for r in identities}) != len(identities):
        raise ValueError("Formal/reserve identities must be source-qualified and distinct")
    selected = [r for r in identities if r["role"] == "formal-proposed"]
    if len(selected) != 6 or any(sum(r["vendor"] == vendor for r in selected) != 3 for vendor in ("Huawei", "Azure")):
        raise ValueError("Exactly three formal identities per vendor are required")
    def digest(row):
        return hashlib.sha256(f"v3-sys|20261022|{row['source']}|{row['app']}".encode()).digest()
    selected.sort(key=digest)
    fault_ids = [r["source"]+"/"+r["app"] for r in selected[:2]]
    if config["faults"]["fragments"] != fault_ids:
        raise ValueError("D8 fault IDs differ from the global hash order")
    rows = []
    def add(identity, kind, arm, delay, repeat, fault):
        raw = digest(identity)
        arrival_fault = "demand3" if fault == "demand3" else "none"
        number = sum(r["kind"] == kind for r in rows)+1
        rows.append(dict(order="", slot_id=f"{kind}-{number:03d}", kind=kind,
            source=identity["source"], app=identity["app"], arm=arm, delay=str(delay), repeat=str(repeat),
            fault=fault, arrival_fault=arrival_fault, seed=str(int.from_bytes(raw[:8], "big")),
            plan_key="v3-"+raw.hex()[:16]+("-demand3" if arrival_fault == "demand3" else ""),
            ticks="240", fault_start_tick="60" if kind == "fault" else "",
            fault_end_tick="120" if kind == "fault" else ""))
    for identity in selected:
        for arm in ARMS:
            for delay, repeats in zip(delays, (1, 1, 2)):
                for repeat in range(1, repeats+1):
                    add(identity, "normal", arm, delay, repeat, "none")
    for identity in selected[:2]:
        for fault in ("startup60", "cap4", "demand3"):
            add(identity, "fault", "HPA-wrapped", 5, 1, fault)
    if len({(r["source"], r["app"], r["arrival_fault"]) for r in rows}) != len({r["plan_key"] for r in rows}):
        raise ValueError("Plan-key collision")
    random.Random(20261023).shuffle(rows)
    for order, row in enumerate(rows, 1):
        row["order"] = str(order)
    return rows


def config_and_plan(root):
    root = Path(root).resolve()
    config = tomllib.loads((root/"configs/v3-sys.toml").read_text("utf-8"))
    return config, build_manifest(config, read_csv(owned(root, config["selection"]["ids"])))


def write_manifest(root=ROOT, path=MANIFEST):
    """Draft planning is allowed; never replace a different existing manifest."""
    _, rows = config_and_plan(root)
    path = owned(root, path)
    if path.exists():
        if read_csv(path, FIELDS) != rows:
            raise ValueError("Different manifest already exists; retain it and resolve before freezing")
        return rows
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def require_frozen(root, config):
    if config["status"] != "FROZEN" or not config.get("freeze_tag") or config["freeze_tag"] == "PENDING":
        raise RuntimeError("Formal reserve blocked: config is not frozen")
    def git(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    ref = "refs/tags/"+config["freeze_tag"]+"^{commit}"
    resolved = git("rev-parse", "--verify", ref)
    if resolved.returncode or not re.fullmatch(r"[0-9a-f]{40,64}", resolved.stdout.strip()):
        raise RuntimeError("Frozen tag does not resolve to a commit")
    revision = resolved.stdout.strip()
    diff = git("diff", "--quiet", revision, "--", *CRITICAL)
    untracked = git("ls-files", "--others", "--exclude-standard", "--", *CRITICAL)
    if diff.returncode or untracked.returncode or untracked.stdout.strip():
        raise RuntimeError("Critical v3 files differ from the frozen tag or contain untracked source")
    if config.get('replay',{}).get('transport_scheme')=='T1':
        from sysbench.calibrate import qualified_transport
        qualified_transport(config,root)
    return revision


@contextmanager
def ledger_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix+".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError("Attempt ledger is locked; inspect an interrupted writer before recovery") from error
    try:
        yield
    finally:
        os.close(descriptor)
        lock.unlink()


def metric(value, name):
    if value in (None, ""):
        return None
    if isinstance(value, bool) or (not isinstance(value, int) and not (isinstance(value, str) and value.isdecimal())):
        raise ValueError(name+" must be a nonnegative integer or explicitly missing")
    value = int(value)
    if value < 0:
        raise ValueError(name+" must be nonnegative")
    return value


def qualifies(row):
    total, inside = metric(row["total_ticks"], "total_ticks"), metric(row["in_range_ticks"], "in_range_ticks")
    if total is not None and inside is not None and inside > total:
        raise ValueError("In-range tick count exceeds observed ticks")
    if row["infrastructure_ok"] not in ("0", "1"):
        raise ValueError("An explicit infrastructure outcome is required")
    return row["infrastructure_ok"] == "1" and total == 240 and inside is not None and inside/total >= .95 and bool(row["log_path"])


def history(manifest, rows):
    slots = {r["slot_id"]: r for r in manifest}
    if len(slots) != 78 or len(manifest) != 78:
        raise ValueError("A complete unique 72+6 manifest is required")
    states, used_run_ids = {}, set()
    for seq, row in enumerate(rows, 1):
        if row["seq"] != str(seq) or row["slot_id"] not in slots:
            raise ValueError("Attempt sequence/slot is invalid")
        if any(row[key] != slots[row["slot_id"]][key] for key in FIELDS):
            raise ValueError("Attempt source/arm/parameters differ from the manifest")
        attempt = metric(row["attempt"], "attempt")
        if attempt not in (0, 1, 2):
            raise ValueError("Only initial attempt plus two reruns are allowed")
        previous = states.get(row["slot_id"])
        event = row["event"]
        if event == "STARTED":
            next_slot = next((r for r in manifest if r["slot_id"] not in states or states[r["slot_id"]]["event"] not in ("VALID", "EXHAUSTED")), None)
            if any(state["event"] == "STARTED" for state in states.values()) or next_slot is None or row["slot_id"] != next_slot["slot_id"]:
                raise ValueError("Single-testbed attempts must follow the fixed manifest order")
            expected = 0 if previous is None else int(previous["attempt"])+1
            if (previous is not None and previous["event"] != "INVALID") or attempt != expected:
                raise ValueError("Attempt skips/reopens an active, valid or exhausted slot")
            if row["run_id"] in used_run_ids or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", row["run_id"]):
                raise ValueError("Attempt run_id must be new and Kubernetes-safe")
            if not row["freeze_tag"] or row["freeze_tag"] == "PENDING" or not re.fullmatch(r"[0-9a-f]{40,64}", row["freeze_commit"]):
                raise ValueError("STARTED must preserve the actual frozen tag/commit")
            used_run_ids.add(row["run_id"])
        else:
            if previous is None or previous["attempt"] != row["attempt"] or previous["run_id"] != row["run_id"]:
                raise ValueError("Outcome does not match its STARTED attempt")
            if any(previous[key] != row[key] for key in ("freeze_tag", "freeze_commit")):
                raise ValueError("Attempt freeze identity changed")
            if event in ("VALID", "INVALID"):
                if previous["event"] != "STARTED" or (event == "VALID") != qualifies(row):
                    raise ValueError("Outcome violates D6 or closes an already closed attempt")
            elif event == "EXHAUSTED":
                if previous["event"] != "INVALID" or attempt != 2:
                    raise ValueError("EXHAUSTED requires the third invalid attempt")
            else:
                raise ValueError("Unknown attempt event")
        states[row["slot_id"]] = row
    return states


def append_events(path, prior, events):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=EVENT_FIELDS)
    if not prior:
        writer.writeheader()
    for i, row in enumerate(events, len(prior)+1):
        row.update(seq=str(i), at_utc=datetime.now(timezone.utc).isoformat())
        writer.writerow(row)
    with path.open("a", encoding="utf-8", newline="") as stream:
        stream.write(output.getvalue())
        stream.flush()
        os.fsync(stream.fileno())


def reserve(root, slot_id, *, manifest_path=MANIFEST, ledger_path=LEDGER, run_id=None):
    root = Path(root).resolve()
    config, expected = config_and_plan(root)
    manifest = read_csv(owned(root, manifest_path), FIELDS)
    if manifest != expected:
        raise ValueError("Manifest differs from the canonical frozen-ID plan")
    revision = require_frozen(root, config)
    slot = next((r for r in manifest if r["slot_id"] == slot_id), None)
    if slot is None:
        raise ValueError("Unknown planned slot")
    path = canonical_ledger(root, ledger_path)
    with ledger_lock(path):
        rows = read_csv(path, EVENT_FIELDS)
        states = history(manifest, rows)
        if any(state["event"] == "STARTED" for state in states.values()):
            raise RuntimeError("One active single-testbed attempt must finish before another reserve")
        next_slot = next((r for r in manifest if r["slot_id"] not in states or states[r["slot_id"]]["event"] not in ("VALID", "EXHAUSTED")), None)
        if next_slot is None or next_slot["slot_id"] != slot_id:
            raise RuntimeError("Reserve must follow the fixed manifest order, including the current slot's reruns")
        if any(r["freeze_tag"] != config["freeze_tag"] or r["freeze_commit"] != revision for r in rows):
            raise ValueError("The attempt ledger is pinned to its first freeze")
        previous = states.get(slot_id)
        if previous and (previous["event"] != "INVALID" or int(previous["attempt"]) >= 2):
            raise RuntimeError("Slot is active, already eligible, or exhausted; no outcome-selected rerun")
        attempt = int(previous["attempt"])+1 if previous else 0
        label = run_id or f"v3-{slot_id}-a{attempt}"
        event = dict(slot, seq="", at_utc="", attempt=str(attempt), event="STARTED", run_id=label,
                     freeze_tag=config["freeze_tag"], freeze_commit=revision, infrastructure_ok="",
                     in_range_ticks="", total_ticks="", reason="RESERVED_BEFORE_EXECUTION", log_path="")
        history(manifest, [*rows, dict(event, seq=str(len(rows)+1))])
        append_events(path, rows, [event])
        return event


def verified_artifact(root, path, started, inside, total):
    """Bind real completed evidence to its slot; reuse the independent analyzer."""
    from sysbench.analyze import analyze
    path = owned(root, path)
    if path.name != started["run_id"]+".jsonl" or not path.is_file():
        raise ValueError("VALID requires the complete, existing reserved-run JSONL")
    records = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    metas = [r for r in records if r.get("type") == "meta"]
    if len(metas) != 1 or records[-1].get("type") != "complete":
        raise ValueError("Completed artifact metadata/closing receipt missing")
    meta = metas[0]
    for key in ("run_id", "source", "app", "arm", "delay", "repeat", "fault", "ticks"):
        if str(meta.get(key)) != str(started[key]):
            raise ValueError("Completed artifact differs from reserved slot: " + key)
    if meta.get("session") is not None:
        raise ValueError("A shakedown artifact cannot be a formal eligible attempt")
    config = tomllib.loads((Path(root)/"configs/v3-sys.toml").read_text("utf-8"))
    if meta.get('transport_scheme','legacy')!=config.get('replay',{}).get('transport_scheme','legacy'):
        raise ValueError("Completed artifact transport differs from the calibrated scheme")
    if meta.get('connection_refresh','none')!=config.get('replay',{}).get('connection_refresh','none'):
        raise ValueError('Completed artifact T1 lifetime policy differs')
    result = analyze(path,capacity_rps=config["calibration"]["capacity_rps"])
    if result["status"] != "RUN_LOG_ANALYZED" or result["schedule_hash_content_verification"] != "VERIFIED" or result["latency"]["status"] != "REQUEST_CSV_RECOMPUTED":
        raise ValueError("Independent real-log/request/plan evidence incomplete")
    if (result["loop_in_range_count"], result["loop_interval_count"]) != (inside, total):
        raise ValueError("Reported timing differs from independently observed intervals")
    return result


def finish(root, slot_id, attempt, *, run_id, infrastructure_ok, in_range_ticks=None,
           total_ticks=None, reason="", log_path="", manifest_path=MANIFEST, ledger_path=LEDGER):
    """Always allow failure registration after STARTED; do not re-gate on clean Git."""
    if not isinstance(infrastructure_ok, bool):
        raise ValueError("infrastructure_ok must be an explicit bool")
    attempt = metric(attempt, "attempt")
    root = Path(root).resolve()
    manifest = read_csv(owned(root, manifest_path), FIELDS)
    path = canonical_ledger(root, ledger_path)
    with ledger_lock(path):
        rows = read_csv(path, EVENT_FIELDS)
        current = history(manifest, rows).get(slot_id)
        if not current or current["event"] != "STARTED" or current["attempt"] != str(attempt) or current["run_id"] != run_id:
            raise ValueError("Finish must match the active source/arm/slot/run attempt")
        if log_path:
            log = owned(root, log_path)
            if log.name not in (run_id+".jsonl", run_id+".partial.jsonl"):
                raise ValueError("Log identity does not match the reserved run")
            log_path = log.relative_to(root).as_posix()
        inside, total = metric(in_range_ticks, "in_range_ticks"), metric(total_ticks, "total_ticks")
        event = dict(current, infrastructure_ok=str(int(infrastructure_ok)),
                     in_range_ticks="" if inside is None else str(inside), total_ticks="" if total is None else str(total),
                     log_path=log_path)
        if qualifies(event):
            try:
                config, expected = config_and_plan(root)
                if expected != manifest or config["freeze_tag"] != current["freeze_tag"] or require_frozen(root, config) != current["freeze_commit"]:
                    raise RuntimeError("Completed attempt no longer matches its original frozen source")
                verified_artifact(root, log_path, current, inside, total)
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                event["infrastructure_ok"] = "0"
                reason = "FREEZE_OR_ARTIFACT_INVALID: "+str(error)+("; "+reason if reason else "")
        valid = qualifies(event)
        event.update(event="VALID" if valid else "INVALID", reason=reason or ("D6_ELIGIBLE" if valid else "INFRASTRUCTURE_OR_TIMING_INVALID"))
        new = [event]
        if not valid and attempt == 2:
            new.append(dict(event, event="EXHAUSTED", reason="INITIAL_PLUS_TWO_RERUNS_EXHAUSTED"))
        history(manifest, [*rows, *(dict(e, seq=str(len(rows)+i)) for i, e in enumerate(new, 1))])
        append_events(path, rows, new)
        return new


def eligible_attempts(root=ROOT, *, manifest_path=MANIFEST, ledger_path=LEDGER):
    manifest = read_csv(owned(root, manifest_path), FIELDS)
    states = history(manifest, read_csv(canonical_ledger(root, ledger_path), EVENT_FIELDS))
    return [states[r["slot_id"]] for r in manifest if r["slot_id"] in states and states[r["slot_id"]]["event"] == "VALID"]


def main(argv=None, *, root=ROOT):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "reserve", "finish"), required=True)
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--ledger", default=LEDGER)
    parser.add_argument("--slot")
    parser.add_argument("--run-id")
    parser.add_argument("--attempt", type=int)
    parser.add_argument("--infrastructure", choices=("ok", "failed"))
    parser.add_argument("--in-range-ticks", type=int)
    parser.add_argument("--total-ticks", type=int)
    parser.add_argument("--reason", default="")
    parser.add_argument("--log", default="")
    args = parser.parse_args(argv)
    if args.stage == "plan":
        rows = write_manifest(root, args.manifest)
        print(f"PLANNED 72+6 slots; metadata only; manifest={args.manifest}; no demand access or execution")
        return rows
    if not args.slot:
        parser.error("--slot is required")
    if args.stage == "reserve":
        event = reserve(root, args.slot, manifest_path=args.manifest, ledger_path=args.ledger, run_id=args.run_id)
        print(f"STARTED slot={args.slot} attempt={event['attempt']} run_id={event['run_id']}")
        return event
    if args.attempt is None or args.run_id is None or args.infrastructure is None:
        parser.error("finish requires --attempt, --run-id, --infrastructure")
    events = finish(root, args.slot, args.attempt, run_id=args.run_id, infrastructure_ok=args.infrastructure == "ok",
        in_range_ticks=args.in_range_ticks, total_ticks=args.total_ticks, reason=args.reason, log_path=args.log,
        manifest_path=args.manifest, ledger_path=args.ledger)
    for event in events:
        print(f"{event['event']} slot={args.slot} attempt={event['attempt']} reason={event['reason']}")
    return events


if __name__ == "__main__":
    main()
