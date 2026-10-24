"""Synthetic checks for the new finite calibration budget and exact D3 selection."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sysbench import calibrate
from sysbench.fragments import _plan_metadata
from sysbench.component import check


def requests(seconds=180, late_tick=None, error_tick=None):
    return [dict(request_id=str(i), offset_ns=str(i*5_000_000_000),
                 written_ns=str(i*5_000_000_000+100_000),
                 completed_ns=str(i*5_000_000_000+(20_000_000 if i == late_tick else 1_000_000)),
                 http_status="200", error="synthetic_error" if i == error_tick else "", write_callbacks="1",
                 pod="p"+str(i%4)) for i in range(seconds//5)]


class CalibrationProperties(unittest.TestCase):
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
