"""D9: eight new Pods while the separate, warmed eight-Pod pool serves 400 RPS.

This calibration may have sixteen total Pods and is not a formal R=8 arm. One
continuous two-minute load session contains a 30s startup probe and, only if its
max is >33.5s, one predeclared 25s probe. Both raw probes are retained.
"""
import json
import os
from pathlib import Path
import time

from sysbench.control import ContractMonitor
from sysbench.runtime import Kube, PodRuntime

WORK=Path(__file__).resolve().parents[1]/"work/v3-sys"


def measure(delay):
    runtime=PodRuntime(Kube(WORK/"kube-access.json"),f"d9-loaded-{delay}",ContractMonitor(96),calibration_background=True)
    result=None
    try:
        for _ in range(8): runtime.create(delay,-1)
        limit=time.monotonic()+70
        while runtime.state()!=(8,0):
            if time.monotonic()>limit: raise TimeoutError("Eight loaded startup Pods not Ready")
            time.sleep(.1)
        starts=[v["ready"]-v["created"] for v in runtime.items.values()]
        runtime.close()
        result=dict(injection_seconds=delay,starts=starts,maximum=max(starts),count=len(starts),
            background_pods=8,background_rate_rps=400,formal_run=False,events=runtime.events)
        (WORK/f"s05-loaded-startup-{delay}.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
        print(json.dumps({k:v for k,v in result.items() if k!="events"}),flush=True)
        return result
    finally:
        try: runtime.close()
        finally:
            if result is None:
                failed=dict(injection_seconds=delay,status="INCOMPLETE",formal_run=False,events=runtime.events)
                (WORK/f"s05-loaded-startup-{delay}-incomplete.json").write_text(json.dumps(failed,indent=2),encoding="utf-8")


if __name__=="__main__":
    os.sched_setaffinity(0,set(range(4,16)))
    first=measure(30)
    selected=30
    if first["maximum"]>33.5:
        time.sleep(1)
        second=measure(25)
        selected=25
        if second["maximum"]>35: raise RuntimeError("Conditional 25s probe also exceeds total startup contract")
    print(json.dumps(dict(selected_injection_seconds=selected,decision="PREDECLARED_D9_STRICT_GT_33_5")),flush=True)
