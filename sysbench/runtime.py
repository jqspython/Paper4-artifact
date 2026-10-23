"""Linux Kubernetes Pod adapter with exact API-call cost boundaries.

Pods use the calibrated image/resources/readiness probe and the existing Service.
Explicit Pod creates/deletes avoid attributing a Deployment's asynchronous calls
to made-up per-Pod creation times. No in-flight create is cancelled by a control
decision. Runtime-cap faults are external mutations, explicitly outside contract.
Credentials stay in ignored work/, never in logs or the public artifact.
"""
import base64
import json
import os
from pathlib import Path
import ssl
import time
import urllib.error
import urllib.request


class Kube:
    def __init__(self, access, *, server="https://paper4-v3-sys-control-plane:6443"):
        access=Path(access)
        config=json.loads(access.read_text("utf-8-sig"))
        cluster=config["clusters"][0]["cluster"]
        user=config["users"][0]["user"]
        cert=access.parent/"runtime-client.crt"
        key=access.parent/"runtime-client.key"
        cert.write_bytes(base64.b64decode(user["client-certificate-data"]))
        key.write_bytes(base64.b64decode(user["client-key-data"]))
        os.chmod(key,0o600)
        context=ssl.create_default_context(cadata=base64.b64decode(cluster["certificate-authority-data"]).decode())
        context.load_cert_chain(cert,key)
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),urllib.request.HTTPSHandler(context=context))
        self.server=server

    def call(self, method, resource, body=None):
        req=urllib.request.Request(self.server+"/api/v1/namespaces/paper4-v3/"+resource,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type":"application/json"},method=method)
        with self.opener.open(req,timeout=3) as response:
            return json.load(response)


def pod(name, owner, delay):
    return {"apiVersion":"v1","kind":"Pod","metadata":{"name":name,
        "labels":{"app":"paper4-cpu-http","paper4-run":owner}},"spec":{
        "terminationGracePeriodSeconds":1,"restartPolicy":"Never","containers":[{
            "name":"cpu-http","image":"paper4-v3-http:feasibility","imagePullPolicy":"Never",
            "env":[{"name":"STARTUP_SLEEP","value":str(delay)}],"ports":[{"containerPort":8080}],
            "resources":{"requests":{"cpu":"250m","memory":"64Mi"},"limits":{"cpu":"250m","memory":"128Mi"}},
            "readinessProbe":{"httpGet":{"path":"/health","port":8080},"periodSeconds":1,
                "timeoutSeconds":1,"failureThreshold":1}}]}}


class PodRuntime:
    def __init__(self, api, owner, monitor, *, calibration_background=False):
        if not owner or not all(c.isalnum() or c=="-" for c in owner):
            raise ValueError("Use an explicit Kubernetes-safe run identity")
        self.api,self.owner,self.monitor=api,owner,monitor
        self.items={}
        self.uncertain={}
        self.events=[]
        self.serial=0
        self.current_tick=-1
        self.cap=8
        existing=self.api.call("GET","pods?labelSelector=app%3Dpaper4-cpu-http")["items"]
        if existing and not calibration_background:
            raise RuntimeError("Service must have no old workload Pods before starting an isolated arm")

    def create(self, delay, tick):
        self.serial+=1
        name=f"{self.owner}-{self.serial:04d}"
        before=time.monotonic()
        event=dict(type="create",pod=name,at=before,command_tick=tick,delay_seconds=delay,accepted=False)
        self.events.append(event)
        self.monitor.pod_requested(name,when=before)
        self.uncertain[name]=before
        event["api_creation_status"]="UNKNOWN_UNTIL_RESPONSE"
        result=self.api.call("POST","pods",pod(name,self.owner,delay))
        event["accepted"]=True
        event["api_creation_status"]="CONFIRMED"
        self.items[name]=dict(created=before,ready=None,deleted=None,delete_confirmed=False,command_tick=tick,seen=False,
                              first_service_tick=None,uid=result["metadata"]["uid"])
        self.uncertain.pop(name)
        return name

    def poll(self):
        found=self.api.call("GET","pods?labelSelector=paper4-run%3D"+self.owner)["items"]
        observed=time.monotonic()
        ready_names=[]
        returned={p["metadata"]["name"] for p in found}
        if any(v["seen"] and v["deleted"] is None and name not in returned for name,v in self.items.items()):
            raise RuntimeError("Previously observed live Pod disappeared; execution contract unverified")
        for item in found:
            name=item["metadata"]["name"]
            if name not in self.items:
                raise ValueError("Unexpected owned Pod")
            saved=self.items[name]
            saved["seen"]=True
            if saved["deleted"] is not None: continue
            if item["metadata"]["uid"] != saved["uid"]:
                raise ValueError("A Pod identity was replaced")
            if item["status"].get("phase") in ("Failed","Succeeded") or item["metadata"].get("deletionTimestamp"):
                raise RuntimeError("Unexpected Pod exit/deletion; preserve invalid run")
            is_ready=any(c["type"]=="Ready" and c["status"]=="True" for c in item["status"].get("conditions",[]))
            if saved["ready"] is not None and not is_ready:
                raise RuntimeError("Ready Pod became NotReady; execution contract unverified")
            if is_ready:
                ready_names.append(name)
                if saved["ready"] is None:
                    saved["ready"]=observed
                    startup=self.monitor.pod_ready(name,when=observed)
                    self.events.append(dict(type="ready",pod=name,at=observed,startup_seconds=startup))
        self.monitor.observe_state(runtime_cap=self.cap,when=observed)
        return ready_names

    def state(self):
        ready=self.poll()
        live=[name for name,v in self.items.items() if v["deleted"] is None]
        return len(ready),len(live)-len(ready)

    def service_start(self, tick):
        ready=self.poll()
        for name in ready:
            saved=self.items[name]
            if saved["first_service_tick"] is None:
                saved["first_service_tick"]=tick
                self.events.append(dict(type="first_service",pod=name,at=time.monotonic(),tick=tick,
                    discrete_lag=tick-saved["command_tick"]))
        return len(ready)

    def delete(self, name, *, external=False):
        item=self.items[name]
        if item["ready"] is None and not external:
            raise ValueError("Cannot cancel an in-flight Pod")
        before=time.monotonic()
        if item["deleted"] is None:
            self.events.append(dict(type="delete",pod=name,at=before,external=external))
            item["deleted"]=before
        else:
            self.events.append(dict(type="cleanup_delete_retry",pod=name,at=before))
        try: self.api.call("DELETE","pods/"+name,{"gracePeriodSeconds":0})
        except urllib.error.HTTPError as error:
            if error.code!=404: raise
        item["delete_confirmed"]=True
        if external: self.monitor.pending.pop(name,None)

    def issue(self, target, delay, tick):
        ready,inflight=self.state()
        current=ready+inflight
        if not 1<=target<=8 or target-current>2:
            raise ValueError("Controller command violates bounded request/rate")
        self.events.append(dict(type="command",at=time.monotonic(),tick=tick,issued_request=target,runtime_cap=self.cap))
        effective=min(target,self.cap)
        if effective>current:
            for _ in range(effective-current): self.create(delay,tick)
        elif effective<current:
            if inflight: raise ValueError("Controller attempted shrink with in-flight Pods")
            ready_names=[n for n,v in self.items.items() if v["deleted"] is None and v["ready"] is not None]
            for name in ready_names[effective:]: self.delete(name)

    def runtime_cap(self, value):
        if value not in (4,8): raise ValueError("Only frozen cap fault or nominal cap")
        self.cap=value
        self.events.append(dict(type="runtime_cap",at=time.monotonic(),runtime_cap=value))
        self.monitor.observe_state(runtime_cap=value)
        live=[n for n,v in self.items.items() if v["deleted"] is None]
        for name in live[value:]: self.delete(name,external=True)

    def close(self):
        errors=[]
        for name,v in self.items.items():
            if v["deleted"] is None or not v["delete_confirmed"]:
                try: self.delete(name,external=True)
                except Exception as error: errors.append(name+": "+str(error))
        for name in tuple(self.uncertain):
            try:
                try: self.api.call("DELETE","pods/"+name,{"gracePeriodSeconds":0})
                except urllib.error.HTTPError as error:
                    if error.code!=404: raise
                self.events.append(dict(type="uncertain_create_cleanup",pod=name,at=time.monotonic(),
                    note="POST_response_uncertain_resource_deleted_or_404;run_remains_invalid"))
                self.uncertain.pop(name)
                self.monitor.pending.pop(name,None)
            except Exception as error: errors.append("UNKNOWN_OWNED_RESOURCE "+name+": "+str(error))
        if errors: raise RuntimeError("Cleanup API errors: "+"; ".join(errors))


def cost_between(events, start, end, *, ready=False):
    """Independent event integration over the declared billing interval."""
    if end<start: raise ValueError("Reversed cost interval")
    creates={e["pod"]:e["at"] for e in events if e["type"]==("ready" if ready else "create")
             and (ready or e.get("accepted",False))}
    deletes={e["pod"]:e["at"] for e in events if e["type"]=="delete"}
    return sum(max(0.,min(end,deletes.get(name,end))-max(start,created)) for name,created in creates.items())
