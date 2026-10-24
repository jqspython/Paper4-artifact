"""Materialize fixed-ID schedules locally; never export fitted demand or arrays."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import tomllib

from sysbench.arrival import poisson_timestamps

ROOT=Path(__file__).resolve().parents[1]


def _plan_metadata(source, app, *, fault, capacity):
    if source not in {"huawei_private2023", "huawei_public2023", "azure2019"} or not app:
        raise ValueError("A fixed source-qualified identity is required")
    if fault not in ("none", "demand3") or isinstance(capacity,bool) or not math.isfinite(capacity) or capacity <= 0:
        raise ValueError("Positive calibrated capacity and declared arrival fault are required")
    digest=hashlib.sha256(f"v3-sys|20261022|{source}|{app}".encode()).digest()
    return dict(source=source,app=app,ticks=240,seed=int.from_bytes(digest[:8],"big"),fault=fault,
        mapping="fitting_prefix_peak_to_4_pods",start_rule="first_240_minutes_of_exploration_test",
        fit_end=4320,val_end=11520,capacity_rps=capacity)


def _verify_plan(path, expected):
    meta=path.with_suffix(".meta.json")
    if path.is_symlink() or meta.is_symlink() or not path.is_file() or not meta.is_file():
        raise ValueError("Both immutable schedule and metadata must exist together")
    metadata=json.loads(meta.read_text("utf-8"))
    if any(metadata.get(k)!=v for k,v in expected.items()):
        raise ValueError("Existing plan identity/mapping differs; never overwrite it")
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(65536),b""): digest.update(chunk)
    if digest.hexdigest()!=metadata.get("schedule_sha256"):
        raise ValueError("Existing immutable schedule hash differs")
    with path.open(encoding="ascii",newline="") as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=["request_id","offset_ns"]: raise ValueError("Schedule schema changed")
        previous=-1; count=0
        for count,row in enumerate(reader,1):
            offset=int(row["offset_ns"])
            if int(row["request_id"])!=count-1 or not 0<=offset<240*5_000_000_000 or offset<previous:
                raise ValueError("Schedule offsets/identities changed")
            previous=offset
    if metadata.get("planned_requests")!=count:
        raise ValueError("Existing schedule count differs from metadata")
    return path


def _materialize(config, source, app, label, *, fault="none", root=ROOT, plan_key=None):
    """Private worker: callers establish reserve/formal authorization first."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,80}",label):
        raise ValueError("Use a local safe plan label")
    expected=_plan_metadata(source,app,fault=fault,capacity=config["calibration"]["capacity_rps"])
    if plan_key is not None: expected["plan_key"]=plan_key
    path=Path(root)/"work/v3-sys"/(label+"-plan.csv")
    meta=path.with_suffix(".meta.json")
    if path.exists() or meta.exists(): return _verify_plan(path,expected)
    import numpy as np
    with np.load(Path(root)/f"data/processed/gate2/{source}.npz",allow_pickle=False) as data:
        ids=[str(x) for x in data["unit_id"]]
        if ids.count(app)!=1: raise ValueError("Fixed app identity is missing/duplicated")
        index=ids.index(app)
        fit_end, val_end=int(data["fit_end"]),int(data["val_end"])
        demand=data["demand"][index]
    if (fit_end,val_end)!=(4320,11520): raise ValueError("Exploration boundaries changed")
    peak=float(np.max(demand[:fit_end]))
    if peak<=0 or not np.isfinite(peak): raise ValueError("Nonpositive fitting-prefix peak")
    rates=(demand[val_end:val_end+240]/peak*(4*config["calibration"]["capacity_rps"])).copy()
    if len(rates)!=240 or not np.all(np.isfinite(rates)) or np.any(rates<0):
        raise ValueError("Invalid fixed first-240-minute fragment")
    if fault=="demand3": rates[60:120]*=3
    values=poisson_timestamps(rates,seed=expected["seed"])
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="ascii",newline="") as out:
        w=csv.writer(out,lineterminator="\n"); w.writerow(("request_id","offset_ns"))
        w.writerows((i,round(t*1e9)) for i,t in enumerate(values))
    metadata=dict(expected,
        schedule_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),planned_requests=len(values),
        access_note="NPZ source body decompressed; only the fixed identity row used; no outcome selection")
    with meta.open("x",encoding="utf-8") as out: json.dump(metadata,out,indent=2,allow_nan=False)
    return _verify_plan(path,expected)


def prepare(session, label, *, fault="none"):
    config=tomllib.loads((ROOT/"configs/v3-sys.toml").read_text("utf-8"))
    ledger=list(csv.DictReader((ROOT/"results/v3-sys-shakedown.csv").open(encoding="utf-8")))
    if isinstance(session,bool) or not isinstance(session,int) or not 1<=session<=len(ledger):
        raise ValueError("A registered reserve session is required")
    entry=ledger[session-1]
    if entry["status"]!="RUNNING": raise ValueError("Register a reserve session before demand access")
    selected=list(csv.DictReader((ROOT/"results/v3-sys-fragment-ids.csv").open(encoding="utf-8")))
    identity=(entry["source"],entry["app"])
    if not any((r["source"],r["app"])==identity and r["role"]=="shakedown-reserve" for r in selected):
        raise ValueError("Shakedown identity is not a reserve")
    return _materialize(config,*identity,label,fault=fault)


def prepare_formal(slot, *, root=ROOT):
    """Only an active, actually reserved frozen-manifest slot can access demand."""
    from sysbench import matrix
    root=Path(root).resolve()
    config,expected=matrix.config_and_plan(root)
    manifest=matrix.read_csv(matrix.owned(root,matrix.MANIFEST),matrix.FIELDS)
    if manifest!=expected: raise ValueError("Formal manifest differs from the fixed ID plan")
    revision=matrix.require_frozen(root,config)
    states=matrix.history(manifest,matrix.read_csv(matrix.owned(root,matrix.LEDGER),matrix.EVENT_FIELDS))
    active=states.get(slot.get("slot_id"))
    if not active or active["event"]!="STARTED" or any(active.get(k)!=v for k,v in slot.items()):
        raise ValueError("Formal materialization requires the active STARTED attempt")
    if active["freeze_commit"]!=revision or active["freeze_tag"]!=config["freeze_tag"]:
        raise ValueError("Formal attempt differs from the frozen source identity")
    metadata=_plan_metadata(active["source"],active["app"],fault=active["arrival_fault"],
                            capacity=config["calibration"]["capacity_rps"])
    expected_key="v3-"+f"{metadata['seed']:016x}"+("-demand3" if metadata["fault"]=="demand3" else "")
    if active["plan_key"]!=expected_key or active["seed"]!=str(metadata["seed"]):
        raise ValueError("Formal plan key/seed differs from the source-qualified hash")
    return _materialize(config,active["source"],active["app"],active["plan_key"],
                        fault=active["arrival_fault"],root=root,plan_key=active["plan_key"])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session",type=int,required=True); p.add_argument("--label",required=True)
    args=p.parse_args()
    print(prepare(args.session,args.label))
