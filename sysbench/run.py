"""One isolated Linux arm. Formal mode is blocked until the v3 config is frozen.

The HTTP process has its own two-core affinity. The control/Ready observations
use the other cores, in the same Linux CLOCK_MONOTONIC domain as the sender.
Failures retain partial logs, terminate offered load, and clean only owned Pods.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import tomllib

from sysbench.control import ContractMonitor, TickController, hpa_request
from sysbench.runtime import Kube, PodRuntime

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/"work/v3-sys"


def check_identity(plan_meta, session, config):
    ids=list(csv.DictReader((ROOT/config["selection"]["ids"]).open(encoding="utf-8")))
    identity=plan_meta["source"],plan_meta["app"]
    if session is not None:
        ledger=list(csv.DictReader((ROOT/"results/v3-sys-shakedown.csv").open(encoding="utf-8")))
        entry=ledger[session-1]
        if entry["status"]!="RUNNING" or (entry["source"],entry["app"])!=identity:
            raise ValueError("Shakedown requires the active registered reserve identity")
        if not any((r["source"],r["app"])==identity and r["role"]=="shakedown-reserve" for r in ids):
            raise ValueError("Formal fragments cannot be used for shakedown")
    elif config["status"]!="FROZEN":
        raise RuntimeError("Formal run blocked: v3 protocol is NOT_FROZEN")
    elif not any((r["source"],r["app"])==identity and r["role"]=="formal-proposed" for r in ids):
        raise ValueError("Formal run must use a frozen formal fragment")


def run(args):
    if os.name!="posix": raise RuntimeError("Run inside the documented Linux container")
    os.sched_setaffinity(0,set(range(4,16)))
    config=tomllib.loads((ROOT/"configs/v3-sys.toml").read_text("utf-8"))
    plan=Path(args.plan)
    plan_meta=json.loads(plan.with_suffix(".meta.json").read_text("utf-8"))
    check_identity(plan_meta,args.session,config)
    if args.session is None:
        if not args.slot_id: raise ValueError("Formal execution requires the reserved fixed slot")
        ledger=list(csv.DictReader((ROOT/"results/v3-sys-attempts.csv").open(encoding="utf-8")))
        related=[r for r in ledger if r["slot_id"]==args.slot_id]
        if not related or related[-1]["event"]!="STARTED" or related[-1]["run_id"]!=args.label:
            raise ValueError("Formal arm is not the active append-only attempt")
        slot=related[-1]
        for key,value in (("source",plan_meta["source"]),("app",plan_meta["app"]),("delay",args.delay),
                          ("repeat",args.repeat),("fault",args.fault)):
            if str(slot[key])!=str(value): raise ValueError("Reserved slot differs from actual arm: "+key)
        canonical={"hpa":"HPA-locked","wrapped-hpa":"HPA-wrapped","conservative-hpa":"HPA-conservative"}[args.arm]
        if slot["arm"]!=canonical: raise ValueError("Reserved slot arm differs")
    if args.initial_replicas!=1 and args.session is None:
        raise ValueError("Formal arms always warm-start with one Pod; eight is a D10-only engineering probe")
    if args.session is None and plan_meta["ticks"]!=240:
        raise ValueError("Formal fragments require all 240 ticks")
    arrival_fault="demand3" if args.fault=="demand3" else "none"
    if plan_meta.get("fault","none")!=arrival_fault:
        raise ValueError("Fault-specific planned arrivals do not match the declared arm")
    duration=plan_meta["ticks"]*5
    fingerprint=hashlib.sha256(plan.read_bytes()).hexdigest()
    if fingerprint!=plan_meta["schedule_sha256"]: raise ValueError("Immutable plan hash mismatch")
    for field,path in (("sender_binary_sha256",WORK/"bin/replay"),("sender_source_sha256",ROOT/"sysbench/replay.go")):
        expected=config["replay"].get(field)
        if args.session is None and (not isinstance(expected,str) or len(expected)!=64
                or any(c not in "0123456789abcdef" for c in expected)):
            raise ValueError("Formal execution requires the runtime-tested sender identity: "+field)
        data=path.read_text("utf-8").encode("utf-8") if field=="sender_source_sha256" else path.read_bytes()
        if expected and hashlib.sha256(data).hexdigest()!=expected:
            raise ValueError("Sender differs from runtime-tested identity: "+field)
    points=list(csv.DictReader(plan.open(encoding="ascii")))
    counts=[0]*plan_meta["ticks"]
    for i,p in enumerate(points):
        if int(p["request_id"])!=i: raise ValueError("Nonconsecutive planned identity")
        counts[int(p["offset_ns"])//5_000_000_000]+=1
    wrapped=args.arm=="wrapped-hpa"
    utilization=.5 if args.arm=="conservative-hpa" else config["controller"]["hpa_utilization"]
    monitor=ContractMonitor(config["calibration"]["capacity_rps"])
    controller=TickController(monitor,wrapped=wrapped)
    runtime=PodRuntime(Kube(WORK/"kube-access.json"),args.label,monitor)
    log=WORK/(args.label+".jsonl")
    partial=log.with_suffix(".partial.jsonl")
    if log.exists() or partial.exists(): raise FileExistsError("Every attempt requires a new immutable run label")
    rows=[]; boundaries=[]; meta=None; child=None; completed=None; event_pos=0; stderr_file=None; primary_error=None
    def save(row):
        rows.append(row)
        with partial.open("a",encoding="utf-8") as out:
            out.write(json.dumps(row,allow_nan=False)+"\n")
    def events():
        nonlocal event_pos
        for event in runtime.events[event_pos:]:
            row=dict(event); row["event_type"]=row.pop("type"); row["type"]="pod_event"; save(row)
        event_pos=len(runtime.events)
    try:
        for _ in range(args.initial_replicas): runtime.create(5,-1)
        until=time.monotonic()+60
        while runtime.state()!=(args.initial_replicas,0):
            if time.monotonic()>until: raise TimeoutError("Initial warm Pod failed to become Ready")
            time.sleep(.1)
        # Identical fixed preheat, outside planned arrivals, retained in paid cost.
        import urllib.request
        for _ in range(10):
            with urllib.request.urlopen("http://paper4-v3-sys-worker:30080/work",timeout=1) as response:
                response.read()
            time.sleep(1)
        command=[str(WORK/"bin/replay"),"-schedule",str(plan),"-target","http://paper4-v3-sys-worker:30080/work",
            "-out",str(WORK/(args.label+"-requests.csv")),"-duration",str(duration),
            "-timeout-ms",str(config["calibration"]["timeout_ms"]),"-slo-ms",str(config["calibration"]["slo_ms"])]
        stderr_file=(WORK/(args.label+"-sender-stderr.log")).open("w",encoding="utf-8")
        child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=stderr_file,text=True,
            preexec_fn=lambda: os.sched_setaffinity(0,{0,2}))
        messages=queue.Queue()
        def receive():
            for line in child.stdout:
                try: messages.put(json.loads(line))
                except Exception as error: messages.put(error)
            messages.put(None)
        thread=threading.Thread(target=receive,daemon=True); thread.start()
        ready_at_start=None; sampled_at=None; end_watchdog=time.monotonic()+duration+35
        while True:
            if time.monotonic()>end_watchdog: raise TimeoutError("Monotonic run watchdog")
            try: event=messages.get(timeout=.1)
            except queue.Empty:
                runtime.poll(); events(); continue
            if event is None: break
            if isinstance(event,Exception): raise event
            if event["type"]=="start":
                if event["schedule_sha256"]!=fingerprint: raise ValueError("Sender plan mismatch")
                # These clocks must share the container host's monotonic epoch.
                if not .5<(event["start_mono_ns"]/1e9-time.monotonic())<3:
                    raise ValueError("Sender/controller monotonic clock mapping failed")
                arm_name={"hpa":"HPA-locked","wrapped-hpa":"HPA-wrapped","conservative-hpa":"HPA-conservative"}[args.arm]
                meta=dict(type="meta",run_id=args.label,arm=arm_name,source=plan_meta["source"],app=plan_meta["app"],
                    delay=args.delay,repeat=args.repeat,session=args.session,schedule_sha256=fingerprint,
                    c_rps=96,slo_ms=18.924,timeout_ms=189.2,ticks=plan_meta["ticks"],fault=args.fault,
                    start_mono=event["start_mono_ns"]/1e9,finish_mono=None,request_csv=args.label+"-requests.csv",
                    schedule_csv=str(plan.resolve().relative_to(WORK.resolve())),initial_replicas=args.initial_replicas,
                    slot_id=args.slot_id,sender_binary_sha256=hashlib.sha256((WORK/"bin/replay").read_bytes()).hexdigest())
                save(meta)
            elif event["type"]=="boundary":
                if meta is None: raise ValueError("Missing sender clock barrier")
                index=event["tick"]
                if index!=len(boundaries): raise ValueError("Missing or duplicate control boundary")
                actual_boundary=time.monotonic()
                if args.fault=="cap4" and index in (60,120): runtime.runtime_cap(4 if index==60 else 8)
                # The first post-boundary Ready query is the operational snapshot;
                # its observation delay is logged, never backdated to ideal time.
                new_ready=runtime.service_start(index)
                new_sampled=time.monotonic()
                boundaries.append(actual_boundary)
                if index:
                    t=index-1
                    ready_now,inflight=runtime.state()
                    delay=60 if args.fault=="startup60" and 60<=index<120 else args.delay
                    b=hpa_request(counts[t],96,utilization)
                    record=controller.finish_tick(t,counts[t],ready_at_start,b,ready_now=ready_now,inflight=inflight,
                        emit=lambda request:runtime.issue(request,delay,t),runtime_cap=runtime.cap)
                    record.update(type="tick",boundary_mono=boundaries[-2],next_boundary_mono=actual_boundary,
                        tick_start_mono=sampled_at,planned_boundary_mono=meta["start_mono"]+t*5,
                        runtime_cap=runtime.cap,issued_request=record["command"],planned_window_count=counts[t],
                        latency_window_p95_ms=None)
                    save(record)
                    if record["error"]:
                        events()  # Preserve actual loss/update and partial API effects.
                        raise RuntimeError(record["error"])
                ready_at_start,sampled_at=new_ready,new_sampled
                events()
            elif event["type"]=="complete":
                completed=event
            else: raise ValueError("Unknown sender event")
        if child.wait(timeout=3)!=0: raise RuntimeError("Sender failed; full stderr retained locally")
        if completed is None or len(boundaries)!=plan_meta["ticks"]+1:
            raise ValueError("Incomplete replay or control run")
        # D1/D3 retain failed/late/missing sends as outcomes, not selective reruns.
        # A complete planned record remains in the SLO denominator as +Inf.
        runtime.close(); events()
        meta["finish_mono"]=time.monotonic()
        windows={r["tick"]:r for r in completed["windows"]}
        for row in rows:
            if row["type"]=="tick": row["latency_window_p95_ms"]=windows[row["index"]]["p95_ms"]
        completed.update(type="complete",planned_requests=completed["planned"])
        save(completed)
        with log.open("w",encoding="utf-8") as out:
            for row in [meta]+[r for r in rows if r is not meta]:
                out.write(json.dumps(row,allow_nan=False)+"\n")
        partial.unlink()
        print(json.dumps(dict(status="MEASURED_NOT_ADJUDICATED",log=str(log),ticks=plan_meta["ticks"])))
    except BaseException as error:
        primary_error=error
        try: save(dict(type="infrastructure_failure",error=type(error).__name__+": "+str(error),at=time.monotonic()))
        except Exception: pass  # A failed evidence sink must not prevent cleanup.
        raise
    finally:
        failures=[]
        if child and child.poll() is None:
            try:
                child.terminate()
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=5)
            except Exception as failure: failures.append("sender: "+str(failure))
        try: runtime.close(); events()
        except Exception as failure: failures.append("owned Pods: "+str(failure))
        if stderr_file:
            try: stderr_file.close()
            except Exception as failure: failures.append("stderr sink: "+str(failure))
        if failures:
            try: save(dict(type="cleanup_failure",error="; ".join(failures),at=time.monotonic()))
            except Exception: pass
            if primary_error is None: raise RuntimeError("Cleanup incomplete: "+"; ".join(failures))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label",required=True)
    parser.add_argument("--plan",required=True)
    parser.add_argument("--arm",choices=("hpa","wrapped-hpa","conservative-hpa"),required=True)
    parser.add_argument("--delay",type=int,choices=(5,15,25,30),required=True)
    parser.add_argument("--repeat",type=int,default=1)
    parser.add_argument("--fault",choices=("none","startup60","cap4","demand3"),default="none")
    parser.add_argument("--session",type=int)
    parser.add_argument("--initial-replicas",type=int,choices=(1,8),default=1)
    parser.add_argument("--slot-id")
    run(parser.parse_args())
