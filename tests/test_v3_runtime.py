"""Changed Kubernetes adapter paths with a fake API; no containers or traffic."""
import unittest
import io
import json
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from sysbench.control import ContractMonitor
from sysbench.runtime import PodRuntime, cost_between, pod
from sysbench.run import refresh_t1, ready_ids
from sysbench.execute import reviewed_pending_operations, maintenance
from sysbench import execute
from sysbench import run
import urllib.error


class ServiceConnectivity(unittest.TestCase):
    def test_sender_clock_is_latched_before_consumer_can_block_on_api_poll(self):
        import queue,threading
        for event_type in ('start','boundary','complete'):
            armed=threading.Event(); messages=queue.Queue()
            run.receive_sender(io.StringIO(json.dumps(dict(type=event_type))+'\n'),messages,armed)
            self.assertTrue(armed.is_set())  # No main-consumer get() has happened yet.
            self.assertEqual(messages.get()['type'],event_type)
        armed=threading.Event()
        run.receive_sender(io.StringIO('{"type":"connection"}\n'),queue.Queue(),armed)
        self.assertFalse(armed.is_set())

    def test_body_timeout_preserves_already_observed_backend_and_http_status(self):
        from contextlib import nullcontext
        clock=[0.]; events=[]
        def body_timeout(_): clock[0]=15.; raise TimeoutError('body stalled')
        response=SimpleNamespace(read=body_timeout,headers={'X-Paper4-Pod':'offline-pod'},status=200)
        with patch.object(run.urllib.request,'urlopen',return_value=nullcontext(response)),\
                patch.object(run.time,'monotonic',side_effect=lambda:clock[0]),self.assertRaises(TimeoutError):
            run.wait_service(emit=events.append)
        self.assertEqual(events[0]['pod'],'offline-pod'); self.assertEqual(events[0]['status'],200)
        self.assertEqual(events[0]['error_type'],'TimeoutError'); self.assertIn('body stalled',events[0]['error'])

    def test_response_completing_after_phase_deadline_never_becomes_setup_success(self):
        from contextlib import nullcontext
        clock=[0.]; events=[]
        def late_read(_): clock[0]=20.; return b'ready\n'
        response=SimpleNamespace(read=late_read,headers={})
        with patch.object(run.urllib.request,'urlopen',return_value=nullcontext(response)),\
                patch.object(run.time,'monotonic',side_effect=lambda:clock[0]),self.assertRaises(TimeoutError):
            run.wait_service(emit=events.append)
        self.assertEqual([e['outcome'] for e in events],['failure'])

    def test_host_keeps_preclock_failure_classification_without_hiding_clocked_failure(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'work') as directory:
            root=Path(directory); work=root/'work/v3-sys'; work.mkdir(parents=True)
            for mode in ('setup','armed-no-meta','clocked'):
                run_id='offline-'+mode
                partial=work/(run_id+'.partial.jsonl')
                records=[dict(type='infrastructure_failure',phase='preclock_setup' if mode=='setup' else 'measurement')]
                if mode=='clocked': records.append(dict(type='meta'))
                partial.write_text('\n'.join(map(json.dumps,records))+'\n',encoding='utf-8')
                slot=dict(run_id=run_id,slot_id='synthetic',attempt='0')
                with patch.object(execute,'maintenance',return_value={}),patch.object(execute.matrix,'config_and_plan',return_value=({},[])),\
                        patch.object(execute.matrix,'reserve',return_value=slot),patch.object(execute,'preflight'),\
                        patch.object(execute,'prepare_formal',return_value=work/'toy-plan.csv'),patch.object(execute,'command',return_value=['offline']),\
                        patch.object(execute.subprocess,'run',return_value=SimpleNamespace(returncode=1)),patch.object(execute,'cleanup_failed'),\
                        patch.object(execute.matrix,'finish',return_value=[]) as finish:
                    execute.execute_slot(root,'synthetic',maintenance_ready=True)
                self.assertEqual(finish.call_args.kwargs['reason'],'HOST_PRECLOCKSETUPERROR' if mode=='setup' else 'HOST_RUNTIMEERROR')

    def test_transient_health_read_timeout_recovers_inside_the_same_setup_budget(self):
        from contextlib import nullcontext
        clock=[0.]; response=SimpleNamespace(read=lambda _:b'ready\n',headers={})
        with patch.object(run.urllib.request,'urlopen',side_effect=[urllib.error.URLError(TimeoutError()),nullcontext(response)]),\
                patch.object(run.time,'monotonic',side_effect=lambda:clock[0]),\
                patch.object(run.time,'sleep',side_effect=lambda delay:clock.__setitem__(0,clock[0]+delay)):
            result=run.wait_service()
            self.assertEqual(result['attempts'],2)
            self.assertEqual(result['connection_refusals'],0)

    def test_refused_service_retries_with_one_shared_deadline(self):
        from contextlib import nullcontext
        clock=[0.]; response=SimpleNamespace(read=lambda _:b'OK',headers={})
        events=[]
        with patch.object(run.urllib.request,'urlopen',side_effect=[urllib.error.URLError(ConnectionRefusedError()),nullcontext(response)]) as opened, \
                patch.object(run.time,'monotonic',side_effect=lambda:clock[0]), \
                patch.object(run.time,'sleep',side_effect=lambda d:clock.__setitem__(0,clock[0]+d)):
            result=run.wait_service(emit=events.append)
        self.assertEqual(result,dict(connection_refusals=1,seconds=.1,attempts=2))
        self.assertEqual(opened.call_count,2)
        self.assertEqual([e['outcome'] for e in events],['failure','success'])

    def test_persistent_refusal_times_out_and_unexpected_errors_are_not_retried(self):
        clock=[0.]
        with patch.object(run.urllib.request,'urlopen',side_effect=urllib.error.URLError(ConnectionRefusedError())) as opened, \
                patch.object(run.time,'monotonic',side_effect=lambda:clock[0]), \
                patch.object(run.time,'sleep',side_effect=lambda d:clock.__setitem__(0,clock[0]+d)),self.assertRaises(TimeoutError):
            run.wait_service(seconds=.05)
        self.assertEqual(clock[0],.05)
        self.assertAlmostEqual(opened.call_args.kwargs['timeout'],.05)
        for error in (urllib.error.URLError(ValueError('unknown')),urllib.error.HTTPError('http://fixture',500,'bad',{},None)):
            with patch.object(run.urllib.request,'urlopen',side_effect=error) as opened, \
                    patch.object(run.time,'sleep') as sleep,self.assertRaises(urllib.error.URLError): run.wait_service()
            self.assertEqual(opened.call_count,1); sleep.assert_not_called()

    def test_fixed_successful_preheat_and_retries_share_one_fifteen_second_budget(self):
        from contextlib import nullcontext
        clock=[0.]; response=SimpleNamespace(read=lambda _:b'OK',headers={})
        with patch.object(run.urllib.request,'urlopen',side_effect=[TimeoutError(),*[nullcontext(response) for _ in range(10)]]) as opened, \
                patch.object(run.time,'monotonic',side_effect=lambda:clock[0]), \
                patch.object(run.time,'sleep',side_effect=lambda d:clock.__setitem__(0,clock[0]+d)):
            result=run.preheat_service()
        self.assertEqual(result['successful_work'],10); self.assertEqual(result['attempts'],11)
        self.assertEqual(opened.call_count,11); self.assertAlmostEqual(result['seconds'],10.1)
        clock[0]=0.
        with patch.object(run.urllib.request,'urlopen',side_effect=TimeoutError()), \
                patch.object(run.time,'monotonic',side_effect=lambda:clock[0]), \
                patch.object(run.time,'sleep',side_effect=lambda d:clock.__setitem__(0,clock[0]+d)),self.assertRaises(TimeoutError):
            run.preheat_service()
        self.assertAlmostEqual(clock[0],15.)


class MaintenanceClassification(unittest.TestCase):
    """Pure fixtures for the changed host gate; no real registry or driver mutations."""
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'work')
        self.addCleanup(self.temporary.cleanup)
        self.base=Path(self.temporary.name); self.temp=self.base/'Temp'; self.temp.mkdir()
        self.windows=self.base/'Windows'
        self.drivers=self.windows/'System32'/'drivers'; self.drivers.mkdir(parents=True)
        self.name='TAOAcceleratorEx64_ev.sys'; self.data=b'artificial cleanup identity'
        self.installed=self.drivers/self.name; self.installed.write_bytes(self.data)
        self.source=self.temp/(self.name+'25-14-52'); self.source.write_bytes(self.data)
        self.expected=hashlib.sha256(self.data).hexdigest()
        self.target=self.windows/'System32'/'spool'/'drivers'/'x64'/'3'/'MXDWDRV.DLL'
        self.target.parent.mkdir(parents=True); self.target.write_bytes(b'fake installed printer')
        self.missing=self.target.parent/'New'/self.target.name
        self.queue=[self.nt(self.source),'',self.nt(self.missing),self.nt(self.target)]
        self.hashes=patch.dict(execute.REVIEWED_CLEANUP,{self.name:self.expected},clear=True)
        self.hashes.start(); self.addCleanup(self.hashes.stop)

    @staticmethod
    def nt(path): return '*1\\??\\'+str(path)

    def classify(self,queue):
        return reviewed_pending_operations(queue,temp_root=self.temp,windows_root=self.windows)

    def test_exact_reviewed_cleanup_and_no_source_residue_are_read_only(self):
        notes=self.classify(self.queue)
        self.assertEqual(len(notes),2)
        self.assertEqual(self.source.read_bytes(),self.data)
        self.assertEqual(self.installed.read_bytes(),self.data)
        self.assertFalse(self.missing.exists())
        self.assertEqual(self.queue[1],'')
        self.assertEqual(self.classify([]),[])

    def test_unknown_path_live_replacement_or_changed_hash_remains_blocked(self):
        bad=[self.queue[:-1],[self.nt(self.temp/'other.sys'),''],
             [self.nt(self.source),self.nt(self.installed)],
             [self.nt(self.missing),self.nt(self.temp/self.target.name)]]
        for queue in bad:
            with self.subTest(queue=queue),self.assertRaises(RuntimeError): self.classify(queue)
        self.source.write_bytes(b'changed')
        with self.assertRaises(RuntimeError): self.classify(self.queue)
        self.source.write_bytes(self.data); self.installed.write_bytes(b'changed installed')
        with self.assertRaises(RuntimeError): self.classify(self.queue)
        self.installed.write_bytes(self.data)
        self.missing.parent.mkdir(); self.missing.write_bytes(b'replacement still pending')
        with self.assertRaises(RuntimeError): self.classify(self.queue)

    def test_windows_update_pending_or_unprotected_window_remains_blocked(self):
        import sys
        from contextlib import nullcontext
        from datetime import datetime,timedelta,timezone
        def opened(hive,key):
            if key.endswith('RebootPending') or key.endswith('RebootRequired'): raise FileNotFoundError
            return nullcontext(key)
        expires=(datetime.now(timezone.utc)+timedelta(days=7)).isoformat()
        fake=SimpleNamespace(HKEY_LOCAL_MACHINE=0,OpenKey=opened,
            QueryValueEx=lambda key,name:(self.queue if name=='PendingFileRenameOperations' else expires,0))
        with patch.dict(sys.modules,{'winreg':fake}),patch.object(execute.os,'name','nt'),\
                patch.object(execute,'update_reboot_required',return_value=False) as update,\
                patch.object(execute,'reviewed_pending_operations',return_value=['synthetic reviewed cleanup']) as review:
            result=maintenance(True)
            self.assertEqual(result['reviewed_delayed_operations'],['synthetic reviewed cleanup'])
            self.assertFalse(result['registry_or_services_changed']); review.assert_called_once_with(self.queue)
            update.return_value=True
            with self.assertRaises(RuntimeError): maintenance(True)
            update.return_value=False; expires=(datetime.now(timezone.utc)-timedelta(days=1)).isoformat()
            with self.assertRaises(RuntimeError): maintenance(True)
            expires=(datetime.now(timezone.utc)+timedelta(days=7)).isoformat()
            fake.OpenKey=lambda *args: nullcontext('pending')
            with self.assertRaises(RuntimeError): maintenance(True)


class API:
    def __init__(self): self.items={}; self.hidden=False
    def call(self,method,resource,body=None):
        if method=="GET": return {"items":[] if self.hidden else list(self.items.values())}
        if method=="POST":
            name=body["metadata"]["name"]
            result={"metadata":{"name":name,"uid":name},"status":{"conditions":[]}}
            self.items[name]=result
            return result
        if method=="DELETE": self.items.pop(resource.split("/")[-1]); return {}
        raise AssertionError(method)
    def ready(self,name,value=True):
        self.items[name]["status"]["conditions"]=[{"type":"Ready","status":"True" if value else "False"}]


class RuntimeProperties(unittest.TestCase):
    def test_t1_refresh_only_on_observed_ready_membership_change(self):
        runtime=SimpleNamespace(items={'a':dict(ready=1,deleted=None),'b':dict(ready=None,deleted=None)})
        child=SimpleNamespace(stdin=io.StringIO())
        previous=ready_ids(runtime)
        self.assertEqual(refresh_t1(child,runtime,previous,1),previous)
        self.assertEqual(child.stdin.getvalue(),'')
        runtime.items['b']['ready']=2
        previous=refresh_t1(child,runtime,previous,2)
        runtime.items['a']['deleted']=3
        refresh_t1(child,runtime,previous,3)
        commands=[json.loads(s) for s in child.stdin.getvalue().splitlines()]
        self.assertEqual([(r['tick'],r['ready_pods']) for r in commands],[(2,['a','b']),(3,['b'])])

    def test_round_two_probe_keeps_period_and_initial_delay(self):
        probe=pod("test","test",5)["spec"]["containers"][0]["readinessProbe"]
        self.assertEqual((probe["timeoutSeconds"],probe["failureThreshold"],probe["periodSeconds"],
                          probe.get("initialDelaySeconds",0)),(2,3,1,0))

    def test_uncertain_post_is_cleaned_by_owned_name_without_claiming_valid_billing(self):
        class UncertainAPI(API):
            def call(self,method,resource,body=None):
                result=super().call(method,resource,body)
                if method=="POST": raise TimeoutError("created server side, response lost")
                return result
        api=UncertainAPI()
        runtime=PodRuntime(api,"uncertain",ContractMonitor(96))
        with self.assertRaises(TimeoutError): runtime.create(5,0)
        self.assertEqual(len(api.items),1)
        runtime.close()
        self.assertFalse(api.items)
        self.assertFalse(runtime.uncertain)
        self.assertFalse(runtime.events[0]["accepted"])
        self.assertEqual(runtime.events[-1]["type"],"uncertain_create_cleanup")

    def test_accepted_not_visible_is_inflight_but_seen_missing_or_ready_regression_is_invalid(self):
        api=API(); clock=[10.]
        with patch("sysbench.runtime.time.monotonic",side_effect=lambda:clock[0]):
            runtime=PodRuntime(api,"synthetic",ContractMonitor(96,clock=lambda:clock[0]))
            name=runtime.create(5,0); api.hidden=True
            self.assertEqual(runtime.state(),(0,1))
            with self.assertRaises(ValueError): runtime.delete(name)
            api.hidden=False; clock[0]=12; self.assertEqual(runtime.state(),(0,1))
            api.hidden=True
            with self.assertRaises(RuntimeError): runtime.state()
            api.hidden=False; api.ready(name); clock[0]=20
            self.assertEqual(runtime.state(),(1,0))
            api.ready(name,False)
            with self.assertRaises(RuntimeError): runtime.state()

    def test_paid_pending_ready_costs_lag_and_bounded_issue(self):
        api=API(); clock=[10.]
        with patch("sysbench.runtime.time.monotonic",side_effect=lambda:clock[0]):
            runtime=PodRuntime(api,"synthetic",ContractMonitor(96,clock=lambda:clock[0]))
            name=runtime.create(5,0)
            with self.assertRaises(ValueError): runtime.issue(4,5,0)
            api.ready(name); clock[0]=20
            self.assertEqual(runtime.service_start(8),1)
            self.assertEqual(runtime.events[-1]["discrete_lag"],8)
            clock[0]=30; runtime.close()
            self.assertEqual(cost_between(runtime.events,0,40),20)
            self.assertEqual(cost_between(runtime.events,0,40,ready=True),10)
            self.assertEqual(cost_between(runtime.events,15,25),10)


if __name__=="__main__": unittest.main()
