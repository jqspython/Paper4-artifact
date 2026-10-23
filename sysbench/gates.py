"""Register (before execution) at most six reserve-only v3 shakedown sessions.

Each gate session has a fixed, explicit subcheck list. No unsuccessful session is
deleted or relabelled as calibration. Raw request data stays in ignored work/.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

from sysbench.arrival import poisson_timestamps

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work/v3-sys"
LEDGER = ROOT / "results/v3-sys-shakedown.csv"


def register(purpose, subchecks):
    prior = list(csv.DictReader(LEDGER.open(encoding="utf-8"))) if LEDGER.exists() else []
    if len(prior) >= 6 or any(r["status"] == "RUNNING" for r in prior):
        raise RuntimeError("Shakedown limit or unfinished session: retain and resolve ledger first")
    ids = list(csv.DictReader((ROOT/"results/v3-sys-fragment-ids.csv").open(encoding="utf-8")))
    reserves = [r for r in ids if r["role"] == "shakedown-reserve"]
    # Same immutable hash rule as the formal selection; never demand/outcome order.
    reserves.sort(key=lambda r: hashlib.sha256(f"v3-sys|{r['seed']}|{r['source']}|{r['app']}".encode()).hexdigest())
    identity = reserves[len(prior)]
    row = dict(session=len(prior)+1, started_utc=datetime.now(timezone.utc).isoformat(),
               source=identity["source"], app=identity["app"], purpose=purpose,
               subchecks=subchecks, status="RUNNING", outcome="PENDING")
    with LEDGER.open("a", newline="", encoding="utf-8") as out:
        w=csv.DictWriter(out,fieldnames=row)
        if not prior: w.writeheader()
        w.writerow(row)
    return row


def finish(session, outcome):
    rows=list(csv.DictReader(LEDGER.open(encoding="utf-8")))
    row=rows[session-1]
    if int(row["session"])!=session or row["status"]!="RUNNING":
        raise ValueError("Only an active registered session can finish")
    row.update(status="COMPLETED",outcome=outcome)
    with LEDGER.open("w",newline="",encoding="utf-8") as out:
        w=csv.DictWriter(out,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)


def plan(path, rate, seconds, *, seed, model="poisson"):
    if seconds <= 0 or seconds % 5 or rate <= 0:
        raise ValueError("Use a positive rate and whole five-second windows")
    values=(poisson_timestamps([rate]*(seconds//5),seed=seed) if model=="poisson"
            else tuple(i/rate for i in range(round(rate*seconds))))
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError("Never overwrite a replay schedule")
    with path.open("w",newline="",encoding="ascii") as out:
        w=csv.writer(out,lineterminator="\n"); w.writerow(("request_id","offset_ns"))
        w.writerows((i,round(t*1e9)) for i,t in enumerate(values))
    return hashlib.sha256(path.read_bytes()).hexdigest(),len(values)


def replay(label, rate, seconds, *, seed, model="poisson", timeout_ms=189.2):
    schedule=WORK/(label+"-plan.csv")
    fingerprint,number=plan(schedule,rate,seconds,seed=seed,model=model)
    command=["docker","run","--rm","--name","paper4-v3-replay","--network","kind",
        "--cpuset-cpus","0,2","--cpus","2","--mount",f"type=bind,source={WORK},target=/bench",
        "--entrypoint","/bench/bin/replay","python:3.11-slim-bookworm@sha256:0a310eeecf4e1f5a0743f9a6520c90c88d089c903ca5fd283f501e3a805f5f89",
        "-schedule","/bench/"+schedule.name,"-target","http://paper4-v3-sys-worker:30080/work",
        "-out","/bench/"+label+"-requests.csv","-duration",str(seconds),"-timeout-ms",str(timeout_ms)]
    result=subprocess.run(command,capture_output=True,text=True,encoding="utf-8",timeout=seconds+40)
    (WORK/(label+"-events.jsonl")).write_text(result.stdout,encoding="utf-8")
    (WORK/(label+"-stderr.log")).write_text(result.stderr,encoding="utf-8")
    if result.returncode: raise RuntimeError("Replay failed; retain raw session evidence: "+label)
    rows=[json.loads(line) for line in result.stdout.splitlines()]
    summary,=[r for r in rows if r["type"]=="complete"]
    if summary["schedule_sha256"]!=fingerprint or summary["planned"]!=number:
        raise ValueError("Plan identity or request-count mismatch")
    return summary


def independent_requests(path):
    """Independent CSV recomputation, not reuse of Go's summary quantiles."""
    rows=list(csv.DictReader(Path(path).open(encoding="utf-8")))
    lateness=sorted((int(r["written_ns"])-int(r["offset_ns"]))/1e6 for r in rows if int(r["written_ns"])>=0)
    latency=sorted((int(r["completed_ns"])-int(r["offset_ns"]))/1e6
        if not r["error"] and int(r["written_ns"])>=0 else math.inf for r in rows)
    wire=sorted((int(r["completed_ns"])-int(r["written_ns"]))/1e6
        if not r["error"] and int(r["written_ns"])>=0 else math.inf for r in rows)
    def q(values,p): return values[math.ceil(len(values)*p)-1] if values else math.nan
    return dict(planned=len(rows),missing=sum(int(r["written_ns"])<0 for r in rows),
        late_p99_ms=q(lateness,.99),latency_p95_ms=q(latency,.95),wire_latency_p95_ms=q(wire,.95),
        retried_requests=sum(int(r["write_callbacks"])>1 for r in rows),
        failed=sum(bool(r["error"]) or int(r["written_ns"])<0 for r in rows))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("register","finish","replay","independent"))
    p.add_argument("--purpose"); p.add_argument("--subchecks"); p.add_argument("--session",type=int)
    p.add_argument("--outcome"); p.add_argument("--label"); p.add_argument("--rate",type=int)
    p.add_argument("--seconds",type=int); p.add_argument("--seed",type=int,default=20261023)
    p.add_argument("--model",choices=("poisson","constant"),default="poisson")
    args=p.parse_args()
    if args.stage=="register": print(register(args.purpose,args.subchecks))
    elif args.stage=="finish": finish(args.session,args.outcome)
    elif args.stage=="independent": print(independent_requests(WORK/(args.label+"-requests.csv")))
    else: print(replay(args.label,args.rate,args.seconds,seed=args.seed,model=args.model))
