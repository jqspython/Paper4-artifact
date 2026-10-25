"""One registered 300s constant-load actuator check, never a workload endpoint."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import tomllib

from sysbench.control import ContractMonitor, TickController
from sysbench.runtime import Kube, PodRuntime
from sysbench.gates import plan
from sysbench.calibrate import ROOT, WORK, IMAGE
from sysbench.analyze import request_windows, verify_t1
from sysbench.run import ready_ids, refresh_t1

LEDGER = ROOT/"results/v3-sys-component.csv"
LABEL = "component-v2-20261024"


def round_two_capacity():
    """The new component is conditional on the same qualified sender and capacity."""
    from sysbench.calibrate import qualified_transport
    config=tomllib.loads((ROOT/"configs/v3-sys.toml").read_text('utf-8'))
    return qualified_transport(config,ROOT)['capacity_rps']


def desired(tick):
    return 5 if tick == 0 or 4 <= tick < 12 else 1


def check(ticks, events, flags, request_execution=False):
    if len(ticks) != 60: raise ValueError("Full sixty-tick component log required")
    plus_two = sum(r["command"]-(r["ready_now"]+r["inflight"]) == 2 for r in ticks)
    blocked = sum(r["request"] < r["ready_now"]+r["inflight"] and r["inflight"] > 0
                  and r["command"] == r["ready_now"]+r["inflight"] for r in ticks)
    downs = [r for r in ticks if r["command"] < r["ready_now"]+r["inflight"]]
    stable = bool(downs)
    for row in downs:
        i = row["index"]
        stable &= row["inflight"] == 0 and i >= 4 and row["command"] == max(
            r["request"] for r in ticks[i-4:i+1])
    creates = {e["pod"]:e for e in events if e["type"] == "create" and e["accepted"]}
    readies = {e["pod"]:e["at"] for e in events if e["type"] == "ready"}
    # Cleanup deletes are outside the control path; normal shrink must only delete Ready Pods.
    cancel = any(e["type"] == "delete" and not e["external"] and
                 (e["pod"] not in readies or e["at"] < readies[e["pod"]]) for e in events)
    additions = {r["index"]:sum(e["command_tick"] == r["index"] for e in creates.values()) for r in ticks}
    active = None; deletions = {}
    for e in events:
        if e["type"] == "command": active = e["tick"]
        if e["type"] == "delete" and not e["external"]:
            deletions[active] = deletions.get(active,0)+1
    actual_actions = all(additions[r["index"]] == max(0,r["command"]-r["ready_now"]-r["inflight"])
        and deletions.get(r["index"],0) == max(0,r["ready_now"]+r["inflight"]-r["command"]) for r in ticks)
    return dict(ticks=len(ticks), plus_two_actions=plus_two, inflight_downscale_blocked_ticks=blocked,
                shrink_actions=len(downs), five_tick_stability=bool(stable), cancelled_inflight=cancel,
                monitor_flags=len(flags), ready_initial=1, created_pods=len(creates),
                accepted_actions_match=actual_actions,request_execution=request_execution,
                passed=plus_two >= 1 and blocked >= 1 and stable and not cancel and not flags and
                       actual_actions and request_execution)


def verify_requests(completed):
    with (WORK/(LABEL+"-requests.csv")).open(encoding="utf-8") as stream: rows=list(csv.DictReader(stream))
    schedule=WORK/(LABEL+"-plan.csv")
    with schedule.open(encoding="ascii") as stream: planned=list(csv.DictReader(stream))
    if len(planned)!=300 or len(rows)!=300 or hashlib.sha256(schedule.read_bytes()).hexdigest()!=completed["schedule_sha256"]:
        return False
    if any((r["request_id"],r["offset_ns"])!=(str(i),str(i*1_000_000_000)) or
           (p["request_id"],p["offset_ns"])!=(str(i),str(i*1_000_000_000)) for i,(r,p) in enumerate(zip(rows,planned))):
        return False
    if "r2" in LABEL and any(int(r.get('transport_id',-1))!=i%256 for i,r in enumerate(rows)): return False
    evidence=request_windows(rows,60)
    return evidence["written"]==300 and evidence["failed"]==0 and evidence["retried_requests"]==0 and evidence["window_counts"]==[5]*60


def worker():
    rows = list(csv.DictReader(LEDGER.open(encoding="utf-8")))
    if len(rows) != 1 or rows[0]["status"] != "RUNNING":
        raise RuntimeError("Component check must be registered before real IO")
    os.sched_setaffinity(0, set(range(4,16)))
    # Forty is a synthetic test unit, not a selected physical capacity; 1 RPS is safely below it.
    capacity=round_two_capacity() if "r2" in LABEL else 40
    monitor = ContractMonitor(capacity)
    controller = TickController(monitor,wrapped=True)
    runtime = PodRuntime(Kube(WORK/"kube-access.json"),LABEL,monitor)
    schedule = WORK/(LABEL+"-plan.csv")
    fingerprint,_ = plan(schedule,1,300,seed=20261024,model="constant")
    child = None; output = []; ticks = []; stderr = None; primary_error = None
    try:
        runtime.create(5,-1)
        limit=time.monotonic()+60
        while runtime.state() != (1,0):
            if time.monotonic()>limit: raise TimeoutError("Initial component Pod")
            time.sleep(.1)
        stderr=(WORK/(LABEL+"-stderr.log")).open("w",encoding="utf-8")
        refresh='refresh' in LABEL
        previous_ready=ready_ids(runtime)
        child=subprocess.Popen([str(WORK/"bin/replay"),"-schedule",str(schedule),"-target",
            "http://paper4-v3-sys-worker:30080/work","-out",str(WORK/(LABEL+"-requests.csv")),
            "-duration","300"]+(["-independent-transports","256"] if "r2" in LABEL else [])+
            (['-refresh-on-stdin'] if refresh else []),stdin=subprocess.PIPE if refresh else None,stdout=subprocess.PIPE,stderr=stderr,text=True,
            preexec_fn=lambda:os.sched_setaffinity(0,{0,2}))
        messages=queue.Queue()
        def receive():
            for line in child.stdout: messages.put(json.loads(line))
            messages.put(None)
        threading.Thread(target=receive,daemon=True).start()
        n_start=None; boundary=[]; limit=time.monotonic()+340
        while True:
            if time.monotonic()>limit: raise TimeoutError("Component watchdog")
            try: event=messages.get(timeout=.1)
            except queue.Empty: runtime.poll(); continue
            if event is None: break
            output.append(event)
            if event["type"] == "boundary":
                i=event["tick"]
                if i != len(boundary): raise ValueError("Consecutive real component boundaries required")
                boundary.append(time.monotonic())
                ready=runtime.service_start(i)
                if i:
                    t=i-1; now,inflight=runtime.state()
                    # Script the proposal interface so constant HTTP traffic exercises both directions.
                    b=desired(t)+controller.value-.125
                    row=controller.finish_tick(t,5,n_start,b,ready_now=now,inflight=inflight,
                        emit=lambda r:runtime.issue(r,15,t))
                    ticks.append(row)
                    if row["error"]: raise RuntimeError(row["error"])
                n_start=ready
                if refresh and i<60: previous_ready=refresh_t1(child,runtime,previous_ready,i,output.append)
        if child.wait(timeout=3) or len(boundary)!=61: raise RuntimeError("Incomplete component sender")
        completed,=[e for e in output if e["type"]=="complete"]
        if refresh:
            started,=[e for e in output if e['type']=='start']
            with (WORK/(LABEL+'-requests.csv')).open(encoding='utf-8') as stream: actual_requests=list(csv.DictReader(stream))
            evidence=[*output,*[dict(r,type='tick') for r in ticks],
                      *[dict(r,type='pod_event',event_type=r['type']) for r in runtime.events]]
            verify_t1(evidence,dict(transport_scheme='T1',connection_refresh='ready_set_change_close_idle',
                                   start_mono=started['start_mono_ns']/1e9),actual_requests)
        result=check(ticks,runtime.events,monitor.flags,verify_requests(completed))
        result.update(http_failed=completed["failed"],schedule_sha256=fingerprint,formal_endpoint=False,
                      load="constant_1_RPS",proposal="scripted_5_at_t0_and_t4_to11_else1",
                      unit_capacity_rps=capacity,physical_capacity_selected="r2" in LABEL)
        if "r2" in LABEL:
            result.update(transport_scheme="T1",sender_binary_sha256=hashlib.sha256((WORK/"bin/replay").read_bytes()).hexdigest(),
                          config_sha256=hashlib.sha256((ROOT/"configs/v3-sys.toml").read_bytes()).hexdigest())
            result['connection_refresh']='ready_set_change_close_idle' if refresh else 'none'
        (WORK/(LABEL+"-result.json")).write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(result),flush=True)
    except BaseException as error:
        primary_error=error; raise
    finally:
        failures=[]
        try:
            if child and child.poll() is None:
                child.terminate()
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=5)
        except Exception as error: failures.append("sender: "+str(error))
        try: runtime.close()
        except Exception as error: failures.append("owned Pods: "+str(error))
        try:
            if stderr: stderr.close()
        except Exception as error: failures.append("stderr: "+str(error))
        try:
            with (WORK/(LABEL+"-events.jsonl")).open("w",encoding="utf-8") as out:
                for row in [dict(type="component_meta",formal_endpoint=False,synthetic=True),*output,
                            *[dict(r,type="component_tick") for r in ticks],
                            *[dict(r,event_type=r["type"],type="pod_event") for r in runtime.events]]:
                    out.write(json.dumps(row,allow_nan=False)+"\n")
        except Exception as error: failures.append("evidence: "+str(error))
        if failures and primary_error is None: raise RuntimeError("Component cleanup/evidence incomplete: "+"; ".join(failures))


def run():
    if LEDGER.exists(): raise FileExistsError("Keep the single component attempt; do not replace it")
    if "r2" in LABEL: round_two_capacity()
    row=dict(label=LABEL,status="RUNNING",load="constant_1_RPS",duration_seconds=300,
             plus_two_actions="",inflight_downscale_blocked_ticks="",shrink_actions="",
             five_tick_stability="",cancelled_inflight="",monitor_flags="",note="not_a_trace_or_endpoint")
    def save():
        with LEDGER.open("w",encoding="utf-8",newline="") as out:
            w=csv.DictWriter(out,fieldnames=row); w.writeheader(); w.writerow(row)
    save()
    try:
        process=subprocess.run(["docker","run","--rm","--name","paper4-driver-"+LABEL,"--network","kind",
            "--label","paper4-v3-run="+LABEL,"--label","paper4-v3-slot=component-"+LABEL,
            "--cpuset-cpus","0,2,4-15","--mount",f"type=bind,source={ROOT},target=/src","--workdir","/src",
            "--entrypoint","python",IMAGE,"-B","-m","sysbench.component","--worker"]+(["--round2"] if "r2" in LABEL else [])+
            (['--refresh-regression'] if 'refresh' in LABEL else []),
            capture_output=True,text=True,encoding="utf-8",timeout=410)
        (WORK/(LABEL+"-host.log")).write_text(process.stdout+"\n"+process.stderr,encoding="utf-8")
        if process.returncode: raise RuntimeError("Component infrastructure failure; evidence retained")
        result=json.loads((WORK/(LABEL+"-result.json")).read_text("utf-8"))
        row.update({k:result[k] for k in row if k in result})
        row["status"]="PASS" if result["passed"] else "FAIL"
        print(json.dumps(result),flush=True)
    except BaseException as error:
        from sysbench.execute import cleanup_failed
        stream=io.StringIO(); cleanup_error=""
        try:
            config=tomllib.loads((ROOT/"configs/v3-sys.toml").read_text("utf-8"))
            cleanup_failed(config,dict(run_id=LABEL,slot_id="component-"+LABEL),stream,root=ROOT)
        except Exception as failure: cleanup_error="; cleanup: "+str(failure)
        row.update(status="INCOMPLETE",note=type(error).__name__+": "+str(error)+cleanup_error)
        try: (WORK/(LABEL+"-cleanup.log")).write_text(stream.getvalue(),encoding="utf-8")
        except OSError: pass
        raise
    finally: save()


def recheck():
    """Stronger offline action/request check of the same attempt; no new traffic."""
    records=[json.loads(line) for line in (WORK/(LABEL+"-events.jsonl")).read_text("utf-8").splitlines()]
    ticks=[r for r in records if r["type"]=="component_tick"]
    events=[dict(r,type=r["event_type"]) for r in records if r["type"]=="pod_event"]
    flags={k:v for r in ticks for k,v in r["first_flags"].items()}
    completed,=[r for r in records if r["type"]=="complete"]
    result=check(ticks,events,flags,verify_requests(completed))
    result.update(analysis="offline_accepted_actions_and_actual_requests",new_run=False)
    target=WORK/(LABEL+"-verified.json")
    if target.exists(): raise FileExistsError("Retain existing verified analysis")
    target.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    with LEDGER.open(encoding="utf-8") as stream: rows=list(csv.DictReader(stream))
    row=rows[0]
    row.update({k:v for k,v in result.items() if k not in ("analysis","new_run","passed")})
    row.update(status="PASS_OFFLINE_RECHECK" if result["passed"] else "FAIL_OFFLINE_RECHECK",note="same_original_attempt_no_new_traffic")
    with LEDGER.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=row); writer.writeheader(); writer.writerow(row)
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",action="store_true"); parser.add_argument("--recheck",action="store_true")
    parser.add_argument("--round2",action="store_true")
    parser.add_argument("--refresh-regression",action="store_true")
    args=parser.parse_args()
    if args.refresh_regression:
        LEDGER=ROOT/'results/v3-sys-component-r2-refresh.csv'
        LABEL='component-r2-refresh-20261025'
    elif args.round2:
        LEDGER=ROOT/"results/v3-sys-component-r2.csv"
        LABEL="component-r2-20261025"
    recheck() if args.recheck else worker() if args.worker else run()
