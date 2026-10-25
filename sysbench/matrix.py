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
from sysbench.arrival import GENERATOR_VERSION


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
    if config.get('execution_amendment'):
        if config.get('schedule_generator',{}).get('version')!=GENERATOR_VERSION:
            raise RuntimeError('Amended execution requires the float64 schedule version')
        verify_amendment_evidence(root,config)
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


def approved_amendment(config):
    amendment=config.get('execution_amendment')
    if amendment is None: return None
    expected=dict(id='d6-float64-20261025',authorization='USER_APPROVED_2026-10-25',
                  void_slot='normal-021',void_attempt=1,extra_slot='fault-002',extra_attempt=3,legacy_rows=11)
    if any(amendment.get(k)!=v for k,v in expected.items()) or amendment.get('activation_tag')!=config.get('freeze_tag'):
        raise ValueError('Only the explicitly approved, named D6 amendment is supported')
    return amendment


def retry_limit(slot_id, amendment=None):
    return 3 if amendment and slot_id==amendment['extra_slot'] else 2


def amendment_reason(amendment, revision):
    return 'USER_APPROVED_D6:'+amendment['id']+':'+amendment['activation_tag']+':'+revision


def verify_amendment_evidence(root,config):
    """Bind the exact stopped prefix and retained artifacts, without rewriting them."""
    root=Path(root).resolve(); amendment=approved_amendment(config)
    if amendment is None: return {}
    raw=canonical_ledger(root,LEDGER).read_bytes()
    prefix=b''.join(raw.splitlines(keepends=True)[:amendment['legacy_rows']+1])
    if hashlib.sha256(prefix).hexdigest()!=amendment.get('legacy_ledger_sha256'):
        raise ValueError('The original stopped ledger prefix changed')
    legacy=list(csv.DictReader(io.StringIO(prefix.decode('utf-8'))))
    if len(legacy)!=11: raise ValueError('The approved prefix must contain its original eleven events')
    manifest=read_csv(root/MANIFEST,FIELDS); states=history(manifest,legacy)
    if (set(states)!={amendment['void_slot'],amendment['extra_slot']}
            or states[amendment['void_slot']]['event']!='VALID'
            or states[amendment['void_slot']]['attempt']!='1'
            or states[amendment['extra_slot']]['event']!='EXHAUSTED'
            or states[amendment['extra_slot']]['attempt']!='2'):
        raise ValueError('The approved legacy outcomes do not match the stopped prefix')
    def ref(tag):
        return subprocess.check_output(['git','rev-parse','--verify','refs/tags/'+tag+'^{commit}'],cwd=root,text=True).strip()
    for tag,revision in {(r['freeze_tag'],r['freeze_commit']) for r in legacy}:
        if ref(tag)!=revision: raise ValueError('Legacy frozen implementation identity changed')
    completed=states[amendment['void_slot']]
    required={r['log_path'] for r in legacy if r['log_path']}
    required.update(('work/v3-sys/'+completed['run_id']+'-requests.csv',
                     'work/v3-sys/'+completed['plan_key']+'-plan.csv',
                     'work/v3-sys/'+completed['plan_key']+'-plan.meta.json'))
    required.update('work/v3-sys/host/'+r['run_id']+'.failure.log' for r in legacy
                    if r['slot_id']==amendment['extra_slot'] and r['event']=='INVALID')
    evidence=amendment.get('evidence',{})
    if not required<=evidence.keys(): raise ValueError('Approved legacy evidence is incomplete')
    for name,digest in evidence.items():
        if hashlib.sha256(owned(root,name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Retained amendment artifact changed: '+name)
    rows=read_csv(canonical_ledger(root,LEDGER),EVENT_FIELDS)
    administrative=[r for r in rows if r['event'] in ('VOID_IMPLEMENTATION','RETRY_AUTHORIZED')]
    if administrative:
        expected=amendment_reason(amendment,ref(amendment['activation_tag']))
        if (len(administrative)!=2 or [r['seq'] for r in administrative]!=['12','13']
                or [r['event'] for r in administrative]!=['VOID_IMPLEMENTATION','RETRY_AUTHORIZED']
                or any(r['reason']!=expected for r in administrative)):
            raise ValueError('Half-applied, duplicate or unbound D6 amendment')
    return {r['run_id']:(r['freeze_tag'],r['freeze_commit']) for r in legacy}


def history(manifest, rows, *, amendment=None):
    if amendment:
        approved_amendment(dict(freeze_tag=amendment.get('activation_tag'),execution_amendment=amendment))
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
        active_amendment=amendment if amendment and seq>amendment['legacy_rows'] else None
        limit=retry_limit(row['slot_id'],active_amendment)
        if attempt is None or not 0<=attempt<=limit:
            raise ValueError("Attempt exceeds the default or explicitly named approved limit")
        previous = states.get(row["slot_id"])
        event = row["event"]
        if event == "STARTED":
            next_slot = next((r for r in manifest if r["slot_id"] not in states or states[r["slot_id"]]["event"] not in ("VALID", "EXHAUSTED")), None)
            if any(state["event"] == "STARTED" for state in states.values()) or next_slot is None or row["slot_id"] != next_slot["slot_id"]:
                raise ValueError("Single-testbed attempts must follow the fixed manifest order")
            expected = 0 if previous is None else int(previous["attempt"])+1
            if (previous is not None and previous["event"] not in ('INVALID','VOID_IMPLEMENTATION','RETRY_AUTHORIZED')) or attempt != expected:
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
                if previous["event"] != "INVALID" or attempt != limit:
                    raise ValueError("EXHAUSTED requires the final allowed invalid attempt")
            elif event in ('VOID_IMPLEMENTATION','RETRY_AUTHORIZED'):
                if not active_amendment:
                    raise ValueError('An explicit prospective D6 amendment is required')
                if event=='VOID_IMPLEMENTATION':
                    correct=(row['slot_id']==amendment['void_slot'] and attempt==1 and previous['event']=='VALID')
                else:
                    correct=(row['slot_id']==amendment['extra_slot'] and attempt==2 and previous['event']=='EXHAUSTED')
                prefix=amendment_reason(amendment,'')
                expected_seq=12 if event=='VOID_IMPLEMENTATION' else 13
                if (not correct or seq!=expected_seq or not row['reason'].startswith(prefix)
                        or not re.fullmatch(r'[0-9a-f]{40,64}',row['reason'][len(prefix):])
                        or any(previous[k]!=row[k] for k in ('log_path','infrastructure_ok','in_range_ticks','total_ticks'))):
                    raise ValueError('The named D6 transition changed facts or lacks its approved source binding')
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
        amendment=approved_amendment(config)
        states = history(manifest, rows,amendment=amendment)
        if any(state["event"] == "STARTED" for state in states.values()):
            raise RuntimeError("One active single-testbed attempt must finish before another reserve")
        if any(state['event']=='EXHAUSTED' for state in states.values()):
            raise RuntimeError('An exhausted slot blocks reserve until the explicit approved amendment is applied')
        next_slot = next((r for r in manifest if r["slot_id"] not in states or states[r["slot_id"]]["event"] not in ("VALID", "EXHAUSTED")), None)
        if next_slot is None or next_slot["slot_id"] != slot_id:
            raise RuntimeError("Reserve must follow the fixed manifest order, including the current slot's reruns")
        current=(config['freeze_tag'],revision)
        foreign=[r for r in rows if (r['freeze_tag'],r['freeze_commit'])!=current]
        if foreign and amendment:
            legacy=verify_amendment_evidence(root,config)
            if any(legacy.get(r['run_id'])!=(r['freeze_tag'],r['freeze_commit']) for r in foreign):
                raise ValueError('An implementation outside the exact approved legacy prefix cannot migrate')
        elif foreign:
            prior={}
            for tag in config.get('execution_compatibility',{}).get('prior_invalid_before_clock_refs',[]):
                sha=subprocess.check_output(['git','rev-parse','--verify','refs/tags/'+tag+'^{commit}'],cwd=root,text=True).strip()
                prior[tag]=sha
            terminals={r['run_id']:r for r in rows}
            for prior_run_id in {r['run_id'] for r in foreign}:
                old=terminals[prior_run_id]
                if old['event']!='INVALID' or prior.get(old['freeze_tag'])!=old['freeze_commit']:
                    raise ValueError('Only declared, immutable, pre-clock INVALID implementations can precede this freeze')
                artifact=owned(root,old['log_path'])
                if old['log_path']!='work/v3-sys/'+prior_run_id+'.partial.jsonl':
                    raise ValueError('Earlier failure must retain its exact partial artifact')
                expected=config.get('execution_compatibility',{}).get('invalid_before_clock_receipts',{}).get(prior_run_id)
                if not expected or hashlib.sha256(artifact.read_bytes()).hexdigest()!=expected:
                    raise ValueError('Earlier failure receipt differs from the explicitly frozen original')
                stderr=owned(root,'work/v3-sys/host/'+prior_run_id+'.stderr.log')
                expected_stderr=config.get('execution_compatibility',{}).get('invalid_before_clock_stderr_receipts',{}).get(prior_run_id)
                if not expected_stderr or hashlib.sha256(stderr.read_bytes()).hexdigest()!=expected_stderr:
                    raise ValueError('Earlier pre-clock traceback differs from the reviewed original')
                records=[json.loads(line) for line in artifact.read_text('utf-8').splitlines() if line.strip()]
                if (not any(r.get('type') in ('infrastructure_failure','host_infrastructure_failure') for r in records)
                        or any(r.get('type') not in ('infrastructure_failure','host_infrastructure_failure','pod_event') for r in records)):
                    raise ValueError('Earlier implementation produced clocked evidence or lacks its retained failure log')
        previous = states.get(slot_id)
        if previous and (previous["event"] not in ('INVALID','VOID_IMPLEMENTATION','RETRY_AUTHORIZED')
                         or int(previous["attempt"]) >= retry_limit(slot_id,amendment)):
            raise RuntimeError("Slot is active, already eligible, or exhausted; no outcome-selected rerun")
        attempt = int(previous["attempt"])+1 if previous else 0
        label = run_id or f"v3-{slot_id}-a{attempt}"
        event = dict(slot, seq="", at_utc="", attempt=str(attempt), event="STARTED", run_id=label,
                     freeze_tag=config["freeze_tag"], freeze_commit=revision, infrastructure_ok="",
                     in_range_ticks="", total_ticks="", reason="RESERVED_BEFORE_EXECUTION", log_path="")
        history(manifest, [*rows, dict(event, seq=str(len(rows)+1))],amendment=amendment)
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
    if meta.get('generator_version')!=GENERATOR_VERSION:
        raise ValueError('Legacy or missing generator version cannot become a current eligible run')
    for key in ("run_id", "source", "app", "arm", "delay", "repeat", "fault", "ticks"):
        if str(meta.get(key)) != str(started[key]):
            raise ValueError("Completed artifact differs from reserved slot: " + key)
    if meta.get("session") is not None:
        raise ValueError("A shakedown artifact cannot be a formal eligible attempt")
    config = tomllib.loads((Path(root)/"configs/v3-sys.toml").read_text("utf-8"))
    from sysbench.fragments import _plan_metadata, _verify_plan
    expected_name=started['plan_key']+'-'+GENERATOR_VERSION+'-plan.csv'
    if meta.get('schedule_csv')!=expected_name:
        raise ValueError('Current VALID requires the exact versioned slot plan')
    plan=owned(root,path.parent/expected_name)
    expected=_plan_metadata(started['source'],started['app'],fault=started['arrival_fault'],
                            capacity=config['calibration']['capacity_rps'])
    expected['plan_key']=started['plan_key']
    plan_meta=json.loads(plan.with_suffix('.meta.json').read_text('utf-8'))
    if plan_meta.get('capacity_rps')!=config['calibration']['capacity_rps']:
        raise ValueError('Plan capacity differs from current frozen calibration')
    _verify_plan(plan,expected)
    if meta.get('schedule_sha256')!=plan_meta['schedule_sha256']:
        raise ValueError('Run and versioned plan fingerprints differ')
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
        candidate=next((r for r in reversed(rows) if r['slot_id']==slot_id),None)
        if not candidate or candidate['event']!='STARTED' or candidate['attempt']!=str(attempt) or candidate['run_id']!=run_id:
            raise ValueError('Finish must match the active source/arm/slot/run attempt')
        # Administrative transitions bind the history to the starting commit,
        # even if the live TOML is damaged after STARTED. Such drift can only
        # finish INVALID; it cannot remove the receipt or reopen a budget.
        if any(r['event'] in ('VOID_IMPLEMENTATION','RETRY_AUTHORIZED') for r in rows):
            if not re.fullmatch(r'[0-9a-f]{40,64}',candidate['freeze_commit']):
                raise ValueError('Malformed starting commit identity')
            frozen=subprocess.check_output(['git','show',candidate['freeze_commit']+':configs/v3-sys.toml'],cwd=root,text=True,encoding='utf-8')
            context=tomllib.loads(frozen)
            if context.get('freeze_tag')!=candidate['freeze_tag']:
                raise ValueError('Starting commit/configuration identity differs')
        else:
            context=tomllib.loads((root/'configs/v3-sys.toml').read_text('utf-8'))
        amendment=approved_amendment(context)
        current = history(manifest, rows,amendment=amendment).get(slot_id)
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
        if not valid and attempt == retry_limit(slot_id,amendment):
            exhausted_reason='APPROVED_NAMED_LIMIT_EXHAUSTED' if amendment and slot_id==amendment['extra_slot'] else 'INITIAL_PLUS_TWO_RERUNS_EXHAUSTED'
            new.append(dict(event, event="EXHAUSTED", reason=exhausted_reason))
        history(manifest, [*rows, *(dict(e, seq=str(len(rows)+i)) for i, e in enumerate(new, 1))],amendment=amendment)
        append_events(path, rows, new)
        return new


def eligible_attempts(root=ROOT, *, manifest_path=MANIFEST, ledger_path=LEDGER):
    config=tomllib.loads((Path(root)/'configs/v3-sys.toml').read_text('utf-8'))
    amendment=approved_amendment(config)
    if amendment: verify_amendment_evidence(root,config)
    manifest = read_csv(owned(root, manifest_path), FIELDS)
    states = history(manifest, read_csv(canonical_ledger(root, ledger_path), EVENT_FIELDS),amendment=amendment)
    selected=[states[r["slot_id"]] for r in manifest if r["slot_id"] in states and states[r["slot_id"]]["event"] == "VALID"]
    if amendment:
        for row in selected:
            with owned(root,row['log_path']).open(encoding='utf-8') as stream:
                meta=next((r for r in (json.loads(line) for line in stream) if r.get('type')=='meta'),{})
            if meta.get('generator_version')!=GENERATOR_VERSION:
                raise ValueError('A legacy VALID cannot enter the amended measurement set')
    return selected


def apply_amendment(root=ROOT):
    """One append-only activation of the user-approved, receipt-bound D6 exception."""
    root=Path(root).resolve(); config,expected=config_and_plan(root)
    revision=require_frozen(root,config); amendment=approved_amendment(config)
    if amendment is None: raise RuntimeError('No approved amendment is configured')
    manifest=read_csv(root/MANIFEST,FIELDS)
    if manifest!=expected: raise ValueError('Fixed manifest changed before amendment')
    path=canonical_ledger(root,LEDGER)
    with ledger_lock(path):
        verify_amendment_evidence(root,config)
        rows=read_csv(path,EVENT_FIELDS); states=history(manifest,rows,amendment=amendment)
        existing=[r for r in rows if r['event'] in ('VOID_IMPLEMENTATION','RETRY_AUTHORIZED')]
        if existing: return dict(status='ALREADY_APPLIED_NO_NEW_EVENTS',events=existing)
        if len(rows)!=11 or any(r['event']=='STARTED' for r in states.values()):
            raise RuntimeError('Amendment requires exactly the stopped prefix and zero active attempts')
        reason=amendment_reason(amendment,revision)
        events=[dict(states[amendment['void_slot']],event='VOID_IMPLEMENTATION',reason=reason),
                dict(states[amendment['extra_slot']],event='RETRY_AUTHORIZED',reason=reason)]
        history(manifest,[*rows,*(dict(e,seq=str(len(rows)+i)) for i,e in enumerate(events,1))],amendment=amendment)
        append_events(path,rows,events)
        return dict(status='APPLIED_TWO_APPEND_ONLY_EVENTS',events=events)


def main(argv=None, *, root=ROOT):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "reserve", "finish", "amend"), required=True)
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
    if args.stage=='amend':
        result=apply_amendment(root); print(result['status']); return result
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
