"""Synthetic checks for the new finite calibration budget and exact D3 selection."""
import csv
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sysbench import calibrate
from sysbench.fragments import _plan_metadata
from sysbench.component import check
from sysbench.report_capacity import component_distribution


def requests(seconds=180, late_tick=None, error_tick=None):
    return [dict(request_id=str(i), offset_ns=str(i*5_000_000_000),
                 written_ns=str(i*5_000_000_000+100_000),
                 completed_ns=str(i*5_000_000_000+(20_000_000 if i == late_tick else 1_000_000)),
                 http_status="200", error="synthetic_error" if i == error_tick else "", write_callbacks="1",
                 pod="p"+str(i%4)) for i in range(seconds//5)]


class CalibrationProperties(unittest.TestCase):
    def test_component_distribution_uses_only_joint_ready_interval_including_zero_pods(self):
        records=[dict(type='start',start_mono_ns=100_000_000_000)]
        records += [dict(type='pod_event',event_type='ready',pod=p,at=t)
                    for p,t in (('a',90),('b',106),('c',108))]
        records += [dict(type='pod_event',event_type='delete',pod='c',at=109)]
        rows=[dict(offset_ns=str(t*1_000_000_000),pod='a') for t in (0,8,9)]
        result=component_distribution(records,rows)
        self.assertEqual(result,dict(maximum_ready=3,planned_requests=1,
                                    pod_counts={'a':1,'b':0,'c':0},missing_or_not_ready=0))

    def test_qualified_sender_is_bound_to_measured_snapshot_and_receipts(self):
        with tempfile.TemporaryDirectory(dir=calibrate.WORK.parent,prefix='r2-provenance-') as directory:
            root=Path(directory); work=root/'work/v3-sys'; (work/'bin').mkdir(parents=True)
            (root/'sysbench').mkdir(); (root/'results').mkdir()
            source='synthetic source\n'; binary=b'synthetic binary'
            source_id=hashlib.sha256(source.encode()).hexdigest(); binary_id=hashlib.sha256(binary).hexdigest()
            text=('\n[platform]\ntick=5\n[controller]\nepsilon=0.05\n[metrics]\nkind="capacity"\n'
                  '[calibration]\ncapacity_rps=0\ncapacity_count_per_tick=0\nslo_ms=18.924\ntimeout_ms=189.2\n'
                  '[replay]\ntransport_scheme="T1"\nfresh_connections=false\n'
                  f'sender_source_sha256="{source_id}"\nsender_binary_sha256="{binary_id}"\n')
            (work/'capacity-r2-config.toml').write_text(text,encoding='utf-8',newline='\n')
            snapshot_id=hashlib.sha256(text.encode()).hexdigest()
            (root/'sysbench/replay.go').write_text(source,encoding='utf-8',newline='\n')
            (work/'bin/replay').write_bytes(binary)
            (work/'capacity-r2-decision.json').write_text(json.dumps(dict(status='PASS',selected_capacity=60,short_runs_used=11)))
            rows=[]
            for i in range(11):
                row=dict(label=f'cal-r2-{i+1:02}',pods='4',per_pod_rps='60',seconds='180',seed='1',
                         transport_scheme='T1',config_sha256=snapshot_id,status='MEASURED')
                rows.append(row)
                events=[dict(type='calibration_driver',sender_binary_sha256=binary_id,schedule_sha256='plan',parameters=row),
                        dict(type='complete',schedule_sha256='plan')]
                (work/(row['label']+'-events.jsonl')).write_text('\n'.join(json.dumps(e) for e in events))
            with (root/'results/v3-sys-capacity-r2.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
            config=calibrate.tomllib.loads(text); config['calibration'].update(capacity_rps=60,capacity_count_per_tick=300)
            self.assertEqual(calibrate.qualified_transport(config,root)['capacity_rps'],60)
            changed=copy.deepcopy(config); changed['replay']['sender_binary_sha256']=hashlib.sha256(b'changed').hexdigest()
            (work/'bin/replay').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'identity changed'): calibrate.qualified_transport(changed,root)
            (work/'bin/replay').write_bytes(binary)
            target=work/'cal-r2-01-events.jsonl'; events=[json.loads(s) for s in target.read_text().splitlines()]
            events[0]['sender_binary_sha256']='different actual binary'
            target.write_text('\n'.join(json.dumps(e) for e in events))
            with self.assertRaisesRegex(ValueError,'Actual calibration sender'): calibrate.qualified_transport(config,root)

    def test_round_two_full_selection_and_late_distribution_stop(self):
        for imbalance_at in (None,5):
            calls=[]
            def fake_point(purpose,rate,**options):
                calls.append((purpose,rate,options))
                return dict(slo_pass=not (purpose=='repeat' and rate==70),fidelity_pass=True,
                            missing_backend=0,backend_max_over_mean=2 if len(calls)==imbalance_at else 1)
            with tempfile.TemporaryDirectory(dir=calibrate.WORK.parent,prefix="r2-selection-") as directory, \
                 patch.object(calibrate,"ROUND",2),patch.object(calibrate,"GRID",(40,50,60,70)), \
                 patch.object(calibrate,"DECISION",Path(directory)/"decision.json"), \
                 patch.object(calibrate,"ledger",side_effect=lambda:calls),patch.object(calibrate,"point",side_effect=fake_point):
                calibrate.run_round_two()
                value=json.loads(calibrate.DECISION.read_text('utf-8'))
                if imbalance_at is None:
                    self.assertEqual((len(calls),value['selected_capacity'],value['status']),(11,60,'PASS'))
                    self.assertEqual(calls[-1][2]['seconds'],300)
                else:
                    self.assertEqual((len(calls),value['status']),(5,'FAILED_DISTRIBUTION_NO_COMPLETE_T2_BUDGET'))

    def test_round_two_integer_window_limits_and_independent_namespace(self):
        pods=["p0","p1","p2","p3"]
        with patch.object(calibrate,"ROUND",2):
            self.assertFalse(calibrate.evaluate(requests(late_tick=0),pods,180,cpu_fraction=.1)["slo_pass"])
            self.assertTrue(calibrate.evaluate(requests(300,late_tick=0),pods,300,combined=True,cpu_fraction=.1)["slo_pass"])
            rows=requests(300,late_tick=0); rows[1]["completed_ns"]=str(5_020_000_000)
            self.assertFalse(calibrate.evaluate(rows,pods,300,combined=True,cpu_fraction=.1)["slo_pass"])
            self.assertFalse(calibrate.evaluate(requests(300,error_tick=40),pods,300,combined=True,cpu_fraction=.1)["slo_pass"])

    def test_component_commands_need_matching_accepted_actions_and_real_requests(self):
        rows=[]; events=[dict(type="create",pod="initial",at=0,accepted=True,command_tick=-1),
                         dict(type="ready",pod="initial",at=1)]
        for i in range(60):
            ready=1 if i<=1 or i>=7 else 3
            inflight=2 if i==1 else 0
            command=3 if i<6 else 1
            rows.append(dict(index=i,ready_now=ready,inflight=inflight,command=command,request=5 if i==0 else 1))
            events.append(dict(type="command",tick=i,at=10+5*i))
        for pod in ("a","b"):
            events += [dict(type="create",pod=pod,at=10,accepted=True,command_tick=0),
                       dict(type="ready",pod=pod,at=18),dict(type="delete",pod=pod,at=40,external=False)]
        events.sort(key=lambda e:e["at"])
        self.assertTrue(check(rows,events,{},True)["passed"])
        self.assertFalse(check(rows,events,{},False)["passed"])
        incomplete=[e for e in events if not (e["type"]=="create" and e["pod"]=="b")]
        self.assertFalse(check(rows,incomplete,{},True)["passed"])

    def test_36_windows_two_percent_means_zero_not_one(self):
        pods = ["p"+str(i) for i in range(4)]
        self.assertTrue(calibrate.evaluate(requests(),pods,180,cpu_fraction=.1)["slo_pass"])
        score = calibrate.evaluate(requests(late_tick=0),pods,180,cpu_fraction=.1)
        self.assertEqual(score["slo_bad_windows"],1)
        self.assertFalse(score["slo_pass"])

    def test_empty_windows_follow_d3_exclusion_rule(self):
        rows=requests(); rows.pop(5)
        for i,r in enumerate(rows): r["request_id"]=str(i)
        score=calibrate.evaluate(rows,["p0","p1","p2","p3"],180,cpu_fraction=.1)
        self.assertEqual(score["slo_windows"],35)
        self.assertTrue(score["slo_pass"])

    def test_final_prefix_and_whole_fidelity_are_distinct(self):
        pods = ["p"+str(i) for i in range(4)]
        # A late failure outside 180s is reported but cannot rewrite the prefix denominator.
        score = calibrate.evaluate(requests(300,error_tick=40),pods,300,combined=True,cpu_fraction=.1)
        self.assertTrue(score["slo_pass"])
        self.assertEqual(score["failed"],1)
        self.assertEqual(score["slo_windows"],36)
        self.assertTrue(score["fidelity_pass"])
        self.assertFalse(calibrate.evaluate(requests(300),pods,300,combined=True,cpu_fraction=.30)["fidelity_pass"])

    def test_zero_backend_and_missing_send_remain_failed_measurements(self):
        rows = requests()
        for r in rows: r.update(pod="",written_ns="-1",error="unsent")
        result = calibrate.evaluate(rows,["p0","p1","p2","p3"],180,cpu_fraction=.1)
        self.assertEqual(result["backend_max_over_mean"],"INF")
        self.assertFalse(result["slo_pass"])
        self.assertFalse(result["fidelity_pass"])
        self.assertIn('"INF"',json.dumps(result,allow_nan=False))

    def test_ledger_limit_and_failed_points_cannot_reset_budget(self):
        with tempfile.TemporaryDirectory(dir=calibrate.WORK.parent,prefix="calibration-property-") as directory, patch.object(calibrate,"LEDGER",Path(directory)/"ledger.csv"), patch.object(calibrate,"DECISION",Path(directory)/"decision.json"):
            self.assertTrue(Path(directory).resolve().is_relative_to(calibrate.WORK.parent.resolve()))
            for i in range(12):
                row = calibrate.register("grid",4,40,180,1,False)
                if i == 0:
                    with self.assertRaises(RuntimeError): calibrate.register("grid",4,40,180,1,False)
                calibrate.finish(row,dict(status="MEASURED",slo_pass=False,note="failed point retained"))
            with self.assertRaises(RuntimeError): calibrate.register("grid",4,40,180,1,False)
            self.assertEqual(len(calibrate.ledger()),12)
            calibrate.DECISION.write_text('{"status":"FAILED_NO_GRID_CAPACITY"}',encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError,"terminal"):
                calibrate.register("grid",4,40,180,1,False)

    def test_plan_metadata_uses_new_capacity_and_rejects_nonpositive(self):
        self.assertEqual(_plan_metadata("azure2019","toy",fault="none",capacity=60)["capacity_rps"],60)
        for v in (0,-1,float("nan"),True):
            with self.assertRaises(ValueError): _plan_metadata("azure2019","toy",fault="none",capacity=v)
