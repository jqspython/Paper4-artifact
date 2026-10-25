"""Host launcher for frozen formal slots; import and --self-check run no traffic.

An operator must declare a protected maintenance window with --maintenance-ready.
This is an operator declaration, not a claim that Windows Update was disabled.
Windows pending-reboot indicators hard-block launch. Sleep inhibition is local
to this process lifetime and is released in finally; no OS policy is changed.
"""
import argparse
from contextlib import contextmanager
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import traceback

from sysbench import matrix
from sysbench.analyze import analyze
from sysbench.fragments import prepare_formal

ROOT=Path(__file__).resolve().parents[1]
IMAGE="python:3.11-slim-bookworm@sha256:0a310eeecf4e1f5a0743f9a6520c90c88d089c903ca5fd283f501e3a805f5f89"
ARM_CLI={"HPA-locked":"hpa", "HPA-wrapped":"wrapped-hpa", "HPA-conservative":"conservative-hpa"}
WATCHDOG_SECONDS=1320


def maintenance(ready):
    if ready is not True:
        raise RuntimeError("MAINTENANCE_REQUIRED: schedule a protected window and explicitly pass --maintenance-ready")
    if os.name=="nt":
        import winreg
        for key in (r"SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending",
                    r"SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired"):
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,key):
                    raise RuntimeError("PENDING_REBOOT: resolve maintenance before formal execution")
            except FileNotFoundError:
                pass
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r"SYSTEM\CurrentControlSet\Control\Session Manager") as key:
                pending,_=winreg.QueryValueEx(key,"PendingFileRenameOperations")
                if pending: raise RuntimeError("PENDING_REBOOT: pending file operations block formal execution")
        except FileNotFoundError:
            pass
    return "OPERATOR_DECLARED_MAINTENANCE_READY; NO_PENDING_REBOOT_DETECTED; UPDATE_POLICY_UNCHANGED"


@contextmanager
def awake():
    changed=False
    set_state=None
    try:
        if os.name=="nt":
            set_state=ctypes.windll.kernel32.SetThreadExecutionState
            set_state.argtypes=[ctypes.c_uint]
            set_state.restype=ctypes.c_uint
            if not set_state(0x80000001):
                raise OSError("Cannot inhibit system sleep for the formal launcher")
            changed=True
        yield
    finally:
        if changed and not set_state(0x80000000):
            raise OSError("Failed to release process sleep inhibition")


def digest(path,*,source_text=False):
    if source_text:
        return hashlib.sha256(Path(path).read_text("utf-8").encode("utf-8")).hexdigest()
    result=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(65536),b""): result.update(chunk)
    return result.hexdigest()


def windows_options():
    return {"creationflags":subprocess.CREATE_NO_WINDOW} if os.name=="nt" else {}


def kubectl(config,root=ROOT):
    return ["kubectl","--kubeconfig",str(Path(root)/"work/v3-sys/kubeconfig"),
        "--context","kind-"+config["platform"]["cluster"],"-n",config["platform"]["namespace"]]


def json_command(command):
    result=subprocess.run(command,capture_output=True,text=True,check=True,timeout=25,**windows_options())
    return json.loads(result.stdout)


def preflight(root,config):
    if config['replay'].get('transport_scheme')=='T1':
        from sysbench.calibrate import qualified_transport
        qualified_transport(config,root)
    deployment=json_command(kubectl(config,root)+["get","deployment","cpu-http","-o","json"])
    if deployment["spec"].get("replicas",1)!=0 or any(
            deployment.get("status",{}).get(k,0)!=0 for k in ("replicas","readyReplicas","availableReplicas")):
        raise RuntimeError("Initial cpu-http Deployment must already be scaled to zero; launcher does not mutate it")
    if json_command(kubectl(config,root)+["get","pods","-l","app=paper4-cpu-http","-o","json"])["items"]:
        raise RuntimeError("Old workload Pods exist; never broadly delete or reuse them")
    for field,path in (("sender_binary_sha256",root/"work/v3-sys/bin/replay"),
                       ("sender_source_sha256",root/"sysbench/replay.go")):
        expected=config["replay"].get(field,"")
        if not re.fullmatch(r"[0-9a-f]{64}",expected) or digest(path,source_text=field=="sender_source_sha256")!=expected:
            raise RuntimeError("Frozen measured sender identity missing/mismatched: "+field)


def command(root,slot,plan):
    relative=Path(plan).resolve().relative_to(Path(root).resolve()).as_posix()
    return ["docker","run","--rm","--name","paper4-driver-"+slot["run_id"],"--network","kind",
        "--label","paper4-v3-run="+slot["run_id"],"--label","paper4-v3-slot="+slot["slot_id"],
        "--cpuset-cpus","0,2,4-15","--mount",f"type=bind,source={root},target=/src",
        "--workdir","/src","--entrypoint","python",IMAGE,"-B","-m","sysbench.run",
        "--label",slot["run_id"],"--slot-id",slot["slot_id"],"--plan","/src/"+relative,
        "--arm",ARM_CLI[slot["arm"]],"--delay",str(slot["delay"]),"--repeat",str(slot["repeat"]),
        "--fault",slot["fault"]]


def verify_result(log,slot,plan,config):
    """Bind complete artifacts before independent analysis; never infer D6 from stdout."""
    log,plan=Path(log),Path(plan)
    if log.name!=slot["run_id"]+".jsonl" or not log.is_file():
        raise ValueError("The complete reserved-run artifact is required")
    records=[json.loads(line) for line in log.read_text("utf-8").splitlines() if line.strip()]
    metas=[row for row in records if row.get("type")=="meta"]
    if len(metas)!=1 or records[-1].get("type")!="complete":
        raise ValueError("Missing unique meta/final complete receipt")
    meta=metas[0]
    for key in ("slot_id","run_id","source","app","arm","delay","repeat","fault","ticks"):
        if str(meta.get(key))!=str(slot[key]): raise ValueError("Actual slot metadata differs: "+key)
    if meta.get("session") is not None or meta.get("initial_replicas")!=1:
        raise ValueError("A reserve/eight-Pod engineering probe cannot be a formal arm")
    if meta.get("request_csv")!=slot["run_id"]+"-requests.csv" or meta.get("schedule_csv")!=plan.name:
        raise ValueError("Run references unexpected request/schedule artifacts")
    plan_meta=json.loads(plan.with_suffix(".meta.json").read_text("utf-8"))
    fingerprint=digest(plan)
    if meta.get("schedule_sha256")!=fingerprint or plan_meta.get("schedule_sha256")!=fingerprint:
        raise ValueError("Run/plan/original-content schedule hashes differ")
    if meta.get("sender_binary_sha256")!=config["replay"]["sender_binary_sha256"]:
        raise ValueError("Actual sender binary differs from the frozen tested binary")
    if meta.get('transport_scheme','legacy')!=config['replay'].get('transport_scheme','legacy'):
        raise ValueError("Actual transport differs from the calibrated frozen scheme")
    if plan_meta.get("capacity_rps") != config["calibration"]["capacity_rps"]:
        raise ValueError("Plan uses a different capacity calibration")
    result=analyze(log,capacity_rps=config["calibration"]["capacity_rps"],expected_ticks=240)
    if (result.get("status")!="RUN_LOG_ANALYZED" or result.get("ticks")!=240 or
            result.get("schedule_hash_content_verification")!="VERIFIED" or
            result.get("latency",{}).get("status")!="REQUEST_CSV_RECOMPUTED"):
        raise ValueError("Independent 240-tick event/request/schedule evidence is incomplete")
    for key in ("run_id","source","app","arm","delay","repeat","schedule_sha256"):
        if str(result.get(key))!=str(meta[key]): raise ValueError("Analyzer identity differs: "+key)
    total,inside=result["loop_interval_count"],result["loop_in_range_count"]
    if isinstance(total,bool) or isinstance(inside,bool) or not isinstance(total,int) or not isinstance(inside,int) or total!=240 or not 0<=inside<=total:
        raise ValueError("D6 requires actual 240 adjacent intervals and an integer in-range count")
    return result


def cleanup_failed(config,slot,stream,*,root=ROOT):
    """Timeout cleanup touches only this attempt container and its exact Pod labels."""
    name="paper4-driver-"+slot["run_id"]
    failures=[]; owned=False
    try:
        inspected=subprocess.run(["docker","inspect","--format","{{json .Config.Labels}}",name],
            capture_output=True,text=True,timeout=25,check=False,**windows_options())
        if inspected.returncode==0:
            labels=json.loads(inspected.stdout)
            owned=labels.get("paper4-v3-run")==slot["run_id"] and labels.get("paper4-v3-slot")==slot["slot_id"]
            if not owned: raise RuntimeError("Container owner labels differ; do not stop it")
        elif "no such" not in inspected.stderr.lower():
            raise RuntimeError("Cannot inspect owned container")
    except Exception as failure:
        failures.append("inspect: "+str(failure)); traceback.print_exc(file=stream)
    if owned:
        for args in (["docker","stop","--time","10",name], ["docker","rm","--force",name]):
            try:
                stopped=subprocess.run(args,capture_output=True,text=True,timeout=30,check=False,**windows_options())
                stream.write((stopped.stdout or "")+(stopped.stderr or ""))
                if stopped.returncode and "no such" not in (stopped.stderr or "").lower():
                    raise RuntimeError("Owned container "+args[1]+" failed")
            except Exception as failure:
                failures.append(args[1]+": "+str(failure)); traceback.print_exc(file=stream)
    try:
        pods=json_command(kubectl(config,root)+["get","pods","-l","paper4-run="+slot["run_id"],"-o","json"])["items"]
    except Exception as failure:
        failures.append("Pod inventory: "+str(failure)); traceback.print_exc(file=stream); pods=[]
    for pod in pods:
        try:
            metadata=pod["metadata"]
            if (metadata.get("labels",{}).get("paper4-run")!=slot["run_id"] or
                    metadata.get("labels",{}).get("app")!="paper4-cpu-http" or
                    not metadata["name"].startswith(slot["run_id"]+"-")):
                raise RuntimeError("Unexpected Pod in owned selector; no broad cleanup permitted")
            subprocess.run(kubectl(config,root)+["delete","pod",metadata["name"],"--ignore-not-found=true",
                           "--wait=true","--timeout=30s"],stdout=stream,stderr=stream,check=True,timeout=35,**windows_options())
        except Exception as failure:
            failures.append("owned Pod: "+str(failure)); traceback.print_exc(file=stream)
    if failures: raise RuntimeError("Owned cleanup incomplete; "+"; ".join(failures))


def execute_slot(root,slot_id,*,maintenance_ready=False):
    """Reserved attempt -> fixed schedule -> isolated driver -> independent finish."""
    root=Path(root).resolve()
    maintenance_note=maintenance(maintenance_ready)
    config,_=matrix.config_and_plan(root)
    slot=matrix.reserve(root,slot_id)
    work=root/"work/v3-sys"; host=work/"host"
    prefix=host/slot["run_id"]
    full=work/(slot["run_id"]+".jsonl"); partial=work/(slot["run_id"]+".partial.jsonl")
    infrastructure=False; inside=total=None; reason="HOST_NOT_COMPLETED"; cleanup_error=None
    launched=False; interrupted=None
    try:
        host.mkdir(parents=True,exist_ok=True)
        with prefix.with_suffix(".launch.json").open("x",encoding="utf-8") as stream:
            json.dump(dict(slot=slot,maintenance=maintenance_note,watchdog_seconds=WATCHDOG_SECONDS),stream,indent=2)
        preflight(root,config)
        plan=prepare_formal(slot,root=root)
        with prefix.with_suffix(".stdout.log").open("x",encoding="utf-8") as out, prefix.with_suffix(".stderr.log").open("x",encoding="utf-8") as err:
            launched=True
            completed=subprocess.run(command(root,slot,plan),cwd=root,stdout=out,stderr=err,
                timeout=WATCHDOG_SECONDS,check=False,**windows_options())
            if completed.returncode: raise RuntimeError("Container driver failed; retain its full stdout/stderr")
        result=verify_result(full,slot,plan,config)
        inside,total=result["loop_in_range_count"],result["loop_interval_count"]
        infrastructure=True; reason="COMPLETE_INDEPENDENT_ARTIFACT_VERIFIED"
    except BaseException as error:
        if not isinstance(error,Exception): interrupted=error
        reason="HOST_"+type(error).__name__.upper()
        try:
            with prefix.with_suffix(".failure.log").open("a",encoding="utf-8") as stream:
                traceback.print_exc(file=stream)
                if launched:
                    try: cleanup_failed(config,slot,stream,root=root)
                    except Exception as failure:
                        traceback.print_exc(file=stream); cleanup_error=failure
        except OSError:
            cleanup_error=error
            if launched:
                # A broken evidence sink must not leave the sender alive.
                try:
                    with open(os.devnull,"w") as sink: cleanup_failed(config,slot,sink,root=root)
                except Exception as failure: cleanup_error=failure
        try:
            if not full.exists() and not partial.exists():
                with partial.open("x",encoding="utf-8") as stream:
                    stream.write(json.dumps(dict(type="host_infrastructure_failure",slot_id=slot_id,
                        run_id=slot["run_id"],reason=reason,driver_logs=prefix.relative_to(root).as_posix()))+"\n")
        except OSError:
            pass  # Missing sink is explicit through empty log_path and infrastructure_ok=False.
    log=full if full.exists() else partial if partial.exists() else None
    events=matrix.finish(root,slot_id,int(slot["attempt"]),run_id=slot["run_id"],
        infrastructure_ok=infrastructure,in_range_ticks=inside,total_ticks=total,reason=reason,
        log_path=log.relative_to(root).as_posix() if log is not None else "")
    if interrupted is not None: raise interrupted
    if cleanup_error is not None:
        raise RuntimeError("Invalid attempt registered; owned-resource cleanup requires inspection") from cleanup_error
    return events


def next_slot(root):
    config,expected=matrix.config_and_plan(root)
    matrix.require_frozen(root,config)
    manifest=matrix.read_csv(matrix.owned(root,matrix.MANIFEST),matrix.FIELDS)
    if manifest!=expected: raise ValueError("Manifest is not the frozen fixed-ID plan")
    states=matrix.history(manifest,matrix.read_csv(matrix.owned(root,matrix.LEDGER),matrix.EVENT_FIELDS))
    if any(r["event"]=="STARTED" for r in states.values()):
        raise RuntimeError("An interrupted/active attempt needs explicit recovery before any new launch")
    if any(r["event"]=="EXHAUSTED" for r in states.values()):
        raise RuntimeError("Matrix incomplete: a slot exhausted initial attempt plus two reruns")
    return next((r for r in manifest if states.get(r["slot_id"],{}).get("event")!="VALID"),None),states


def self_check():
    """Minimal constructed command/identity checks; no trace, Docker or registry IO."""
    from unittest.mock import patch
    from sysbench import fragments
    from contextlib import nullcontext
    import copy
    import io
    import tempfile
    from types import SimpleNamespace
    import numpy as np
    sample=dict(slot_id="normal-001",run_id="toy-a0",source="azure2019",app="toy",
        arm="HPA-wrapped",delay="30",repeat="2",fault="none",ticks="240")
    args=command(ROOT,sample,ROOT/"work/v3-sys/toy-plan.csv")
    assert args[args.index("--arm")+1]=="wrapped-hpa" and args[args.index("--slot-id")+1]=="normal-001"
    assert "--session" not in args and "--env" not in args and WATCHDOG_SECONDS==1320
    platform={"platform":{"cluster":"toy","namespace":"toy"}}
    access=kubectl(platform,ROOT)
    assert access[access.index("--kubeconfig")+1]==str(ROOT/"work/v3-sys/kubeconfig")
    ordinary=fragments._plan_metadata("azure2019","toy",fault="none",capacity=96)
    fault=fragments._plan_metadata("azure2019","toy",fault="demand3",capacity=96)
    assert ordinary["seed"]==fault["seed"] and ordinary["fault"]!=fault["fault"]
    try: maintenance(False)
    except RuntimeError: pass
    else: raise AssertionError("Maintenance acknowledgement absent")
    with patch.object(matrix,"reserve",side_effect=RuntimeError("NOT_FROZEN")), patch.object(
            matrix,"config_and_plan",return_value=({},[])),patch(
            __name__+".maintenance",return_value="synthetic"),patch(__name__+".prepare_formal") as prepare:
        try: execute_slot(ROOT,"normal-001",maintenance_ready=True)
        except RuntimeError: pass
        else: raise AssertionError("Frozen gate bypassed")
        prepare.assert_not_called()
    with patch.object(matrix,"config_and_plan",return_value=({},[])),patch.object(
            matrix,"read_csv",return_value=[]),patch.object(matrix,"require_frozen",side_effect=RuntimeError("NOT_FROZEN")),patch.object(
            fragments,"_materialize") as materialize:
        try: fragments.prepare_formal(sample)
        except RuntimeError: pass
        else: raise AssertionError("Formal freeze gate bypassed")
        materialize.assert_not_called()
    fake_state=SimpleNamespace(kernel32=SimpleNamespace(SetThreadExecutionState=lambda flag: calls.append(flag) or 1))
    calls=[]
    with patch.object(os,"name","nt"),patch.object(ctypes,"windll",fake_state,create=True):
        try:
            with awake(): raise ValueError("synthetic interruption")
        except ValueError: pass
    assert calls==[0x80000001,0x80000000]
    cleanup_calls=[]
    def fake_cleanup(args,**kwargs):
        cleanup_calls.append(args)
        if args[:2]==["docker","inspect"]:
            return SimpleNamespace(returncode=0,stdout=json.dumps({"paper4-v3-run":sample["run_id"],
                "paper4-v3-slot":sample["slot_id"]}),stderr="")
        if args[:2]==["docker","stop"]: raise subprocess.TimeoutExpired("synthetic stop",30)
        return SimpleNamespace(returncode=0,stdout="",stderr="")
    pod={"metadata":{"name":sample["run_id"]+"-0001",
        "labels":{"app":"paper4-cpu-http","paper4-run":sample["run_id"]}}}
    with patch.object(subprocess,"run",side_effect=fake_cleanup),patch(__name__+".json_command",return_value={"items":[pod]}):
        try: cleanup_failed(platform,sample,io.StringIO(),root=ROOT)
        except RuntimeError: pass
        else: raise AssertionError("Cleanup timeout was not reported")
    assert any(a[:2]==["docker","rm"] for a in cleanup_calls)
    assert any("delete" in a and "pod" in a for a in cleanup_calls)
    fixture=tempfile.TemporaryDirectory(dir=ROOT/"work",prefix="v3-launcher-property-")
    toy=Path(fixture.name).resolve()
    assert toy.is_relative_to((ROOT/"work").resolve())  # Verify exact owned cleanup target.
    try:
        source=toy/"sender.go"; source.write_bytes(b"one\r\ntwo\n")
        assert digest(source,source_text=True)==hashlib.sha256(b"one\ntwo\n").hexdigest()
        assert digest(source)!=digest(source,source_text=True)
        demand=np.zeros((1,11760)); demand[0,:4320]=100.; demand[0,11520:]=.05
        data=dict(unit_id=np.array(["toy"]),fit_end=4320,val_end=11520,demand=demand)
        config={"calibration":{"capacity_rps":96}}
        with patch.object(np,"load",return_value=nullcontext(data)),patch.object(
                fragments,"poisson_timestamps",return_value=(.1,301.1)) as poisson:
            normal=fragments._materialize(config,"azure2019","toy","v3-toy",root=toy)
            normal_rates=poisson.call_args.args[0].copy(); seed=poisson.call_args.kwargs["seed"]
            fragments._materialize(config,"azure2019","toy","v3-toy-demand3",fault="demand3",root=toy)
            changed_rates=poisson.call_args.args[0]
            assert poisson.call_args.kwargs["seed"]==seed
            np.testing.assert_allclose(normal_rates,np.full(240,.192))
            np.testing.assert_allclose(changed_rates[60:120],normal_rates[60:120]*3)
            np.testing.assert_array_equal(changed_rates[:60],normal_rates[:60])
            np.testing.assert_array_equal(changed_rates[120:],normal_rates[120:])
        before=normal.read_bytes()
        with patch.object(np,"load",side_effect=AssertionError("Cached plan must not reread demand")):
            assert fragments._materialize(config,"azure2019","toy","v3-toy",root=toy)==normal
            meta=normal.with_suffix(".meta.json")
            wrong=json.loads(meta.read_text("utf-8")); wrong["seed"]+=1
            meta.write_text(json.dumps(wrong),encoding="utf-8")
            try: fragments._materialize(config,"azure2019","toy","v3-toy",root=toy)
            except ValueError: pass
            else: raise AssertionError("Changed cached metadata accepted")
        assert normal.read_bytes()==before
    finally:
        fixture.cleanup()
    # Incomplete evidence must not become a VALID result; full artifact checks
    # are exercised by the analyzer/matrix suites with constructed fixtures.
    bad=copy.copy(sample); bad["arm"]="unknown"
    try: command(ROOT,bad,ROOT/"work/v3-sys/toy-plan.csv")
    except KeyError: pass
    else: raise AssertionError("Unknown arm accepted")
    print("self_check=PASS: arm/slot binding, maintenance/freeze gates, sleep-finally, synthetic scaling/fault/seed, immutable cache; no real trace/traffic")


def main(argv=None,*,root=ROOT):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--slot"); mode.add_argument("--all",action="store_true")
    mode.add_argument("--self-check",action="store_true")
    parser.add_argument("--maintenance-ready",action="store_true")
    args=parser.parse_args(argv)
    if args.self_check: self_check(); return
    maintenance(args.maintenance_ready)
    next_slot(root)  # Frozen source gate precedes sleep changes and any demand IO.
    with awake():
        if args.slot:
            events=execute_slot(root,args.slot,maintenance_ready=args.maintenance_ready)
            print(events[-1]["event"],args.slot,"attempt",events[-1]["attempt"])
            return
        while True:
            slot,states=next_slot(root)
            if slot is None:
                total=sum(int(r["total_ticks"]) for r in states.values())
                inside=sum(int(r["in_range_ticks"]) for r in states.values())
                print(json.dumps(dict(status="ALL_78_ELIGIBLE_NO_REOPEN",in_range_ticks=inside,total_ticks=total,
                    pooled_fraction=inside/total,pooled_pass=inside/total>=.99,
                    failure_action="overall_failure_only_never_retry_VALID")))
                return
            events=execute_slot(root,slot["slot_id"],maintenance_ready=args.maintenance_ready)
            print(events[-1]["event"],slot["slot_id"],"attempt",events[-1]["attempt"])


if __name__=="__main__":
    main()
