"""Artificial event/request logs exercise offline analysis, never system evidence."""
import copy
import hashlib
import unittest

from sysbench.analyze import analyze_matrix, analyze_run, monitor_evidence, request_windows
from sysbench.control import ContractMonitor, TickController


def fixture(counts=(1, 1), *, arm="HPA-wrapped", intervals=None, request_errors=()):
    intervals = intervals or [5.] * len(counts)
    monitor = ContractMonitor(96)
    controller = TickController(monitor, wrapped=arm == "HPA-wrapped")
    meta = dict(type="meta", run_id="artificial", arm=arm, source="source", app="app",
                delay=30, repeat=1, schedule_sha256="a"*64, c_rps=96,
                slo_ms=18.924, timeout_ms=189.2, start_mono=100., finish_mono=101.+sum(intervals))
    events = [dict(type="pod_event", event_type="create", pod="initial", at=90., accepted=True, command_tick=-1),
              dict(type="pod_event", event_type="ready", pod="initial", at=92.)]
    ticks, requests = [], []
    begin = 100.
    for i, (n, interval) in enumerate(zip(counts, intervals)):
        end = begin+interval
        row = controller.finish_tick(i, n, 1, 1., ready_now=1, inflight=0, emit=lambda _: None, when=end)
        failed = i in request_errors
        row.update(type="tick", tick_start_mono=begin+.25, boundary_mono=begin,
                   next_boundary_mono=end, runtime_cap=8, issued_request=row["command"],
                   planned_window_count=n, latency_window_p95_ms=("INF" if failed else 1.) if n else None)
        ticks.append(row)
        events.append(dict(type="pod_event", event_type="command", at=end+.05, tick=i,
                           issued_request=row["command"], runtime_cap=8))
        for j in range(n):
            offset = i*5_000_000_000+j*1000
            requests.append(dict(request_id=str(len(requests)), offset_ns=str(offset),
                written_ns=str(offset+100_000), completed_ns=str(offset+1_000_000),
                http_status="500" if failed else "200", error="HTTP_500" if failed else "", write_callbacks="1"))
        begin = end
    events.append(dict(type="pod_event", event_type="delete", pod="initial", at=begin+.5))
    complete = dict(type="complete", planned_requests=sum(counts), written=sum(counts),
                    failed=sum(n for i, n in enumerate(counts) if i in request_errors), missing_write=0)
    return [meta, *ticks, *events, complete], requests


class RealLogAnalysis(unittest.TestCase):
    def test_d3_rebuild_infinite_failures_empty_windows_and_creation_billing(self):
        records, requests = fixture((1, 0, 1), request_errors=(2,))
        result = analyze_run(records, expected_ticks=3, request_rows=requests)
        self.assertEqual(result["latency"]["window_p95"], [1., None, "INF"])
        self.assertEqual(result["latency"]["slo_violation_rate"], .5)
        self.assertEqual(result["latency"]["empty_windows"], 1)
        self.assertEqual(result["paid_replica_seconds"], 25.5)  # create90 -> delete115.5
        self.assertEqual(result["ready_replica_seconds"], 23.5)
        self.assertNotEqual(result["paid_replica_seconds"], 3*5)  # Never n_start*5.
        self.assertEqual(result["engineering_matrix"], "NOT_ASSESSED")

    def test_d3_uses_planned_window_and_deadline_including_unsent_and_tail(self):
        rows = [dict(request_id="0", offset_ns="4999900000", written_ns="5000000000",
                     completed_ns="5000900000", http_status="200", error="", write_callbacks="1"),
                dict(request_id="1", offset_ns="5000000000", written_ns="-1", completed_ns="-1",
                     http_status="0", error="not_sent", write_callbacks="0"),
                dict(request_id="2", offset_ns="5100000000", written_ns="5100100000", completed_ns="5300000000",
                     http_status="200", error="", write_callbacks="1")]
        result = request_windows(rows, 2)
        self.assertEqual(result["window_counts"], [1, 2])
        self.assertEqual(result["window_p95"][0], 1.)
        self.assertEqual(result["failed"], 2)
        self.assertEqual(result["missing_write"], 1)

    def test_unwrapped_na_and_missing_request_csv_remain_explicit(self):
        records, _ = fixture(arm="HPA-locked")
        result = analyze_run(records, expected_ticks=2)
        self.assertEqual(result["certificate_applicability"], "NOT_APPLICABLE")
        self.assertEqual(result["latency"]["status"], "NOT_ASSESSED")

    def test_independent_cap_truth_does_not_depend_on_online_flags(self):
        records, _ = fixture()
        records.insert(-1, dict(type="pod_event", event_type="runtime_cap", at=103., runtime_cap=4))
        result = analyze_run(records, expected_ticks=2)
        self.assertEqual(result["independent_truth"], {"runtime_cap": 103.})
        self.assertIsNone(result["monitor"]["missed_detection"])
        self.assertIn("NOT_ASSESSED", result["monitor"]["status"])
        self.assertFalse(result["monitor"]["false_positive"])
        self.assertIsNone(result["monitor"]["detected_within_10_ticks"])
        self.assertEqual(result["certificate_applicability"], "OUTSIDE_OBSERVED_CONTRACT")

    def test_envelope_and_actual_issued_request_are_independent_of_raw_saturation(self):
        records, _ = fixture((3841,)+(1,)*12)
        records[1]["issued_request"] = 9
        for row in records:
            if row.get("event_type") == "command" and row["tick"] == 0:
                row["issued_request"] = 9
        result = analyze_run(records, expected_ticks=13)
        self.assertEqual(result["independent_truth"]["demand_envelope"], 105.)
        self.assertEqual(result["independent_truth"]["request_above_maximum"], 105.05)
        self.assertTrue(result["monitor"]["detected_within_10_ticks"])
        records, _ = fixture()
        records[1].update(unclipped_request=100., saturated=True)
        self.assertEqual(analyze_run(records, expected_ticks=2)["independent_truth"], {})

    def test_false_positive_and_strict_startup_truth_include_unready_pod(self):
        records, _ = fixture()
        for row in records:
            if row["type"] == "tick":
                row["first_flags"] = {"startup_bound": dict(observed_at=105., tick=0)}
        self.assertTrue(analyze_run(records, expected_ticks=2)["monitor"]["false_positive"])
        records, _ = fixture((1,)*16)
        records[-1:-1] = [dict(type="pod_event", event_type="create", pod="slow", at=80., accepted=True, command_tick=-1),
                         dict(type="pod_event", event_type="ready", pod="slow", at=116.),
                         dict(type="pod_event", event_type="delete", pod="slow", at=180.5)]
        for row in records:
            if row["type"] == "tick" and row["index"] >= 3:
                row["first_flags"] = {"startup_bound": dict(observed_at=120., tick=3)}
            if row["type"] == "tick" and row["index"] >= 4:
                row["n_start"] = 2
        result = analyze_run(records, expected_ticks=16)
        self.assertEqual(result["independent_truth"]["startup_bound"], 115.)
        self.assertEqual(result["monitor"]["detection_delay_seconds"], 5.)
        self.assertTrue(result["monitor"]["detected_within_10_ticks"])
        records, _ = fixture()
        records.insert(-1, dict(type="pod_event", event_type="create", pod="pending", at=70.,
                               accepted=True, command_tick=-1))
        records.insert(-1, dict(type="pod_event", event_type="delete", pod="pending", at=110.5))
        self.assertEqual(analyze_run(records, expected_ticks=2)["independent_truth"]["startup_bound"], 105.)

    def test_startup_exact_35_is_within_contract_and_late_latch_misses(self):
        records, _ = fixture()
        for row in records:
            if row.get("event_type") == "create":
                row["at"] = 57.
        self.assertNotIn("startup_bound", analyze_run(records, expected_ticks=2)["independent_truth"])
        records, _ = fixture((1,)*12)
        records.insert(-1, dict(type="pod_event", event_type="runtime_cap", at=100., runtime_cap=4))
        for row in records:
            if row["type"] == "tick" and row["index"] >= 10:
                row["first_flags"] = {"runtime_cap": dict(observed_at=155., tick=10)}
        result = analyze_run(records, expected_ticks=12)
        self.assertTrue(result["monitor"]["missed_detection"])
        self.assertEqual(result["monitor"]["detection_delay_seconds"], 55.)

    def test_detection_uses_actual_boundary_ticks_not_fixed_50_seconds(self):
        for interval, elapsed_ticks, expected in ((6., 9, True), (4., 11, False)):
            boundaries = [100.+i*interval for i in range(15)]
            flag = 100.+elapsed_ticks*interval
            result = monitor_evidence({"runtime_cap": 100.}, {"runtime_cap": {"observed_at": flag}}, boundaries)
            self.assertEqual(result["detected_within_10_ticks"], expected)
            self.assertEqual(result["detection_delay_ticks"], elapsed_ticks)
        short = monitor_evidence({"runtime_cap": 109.}, {}, [100., 105., 110.])
        self.assertIsNone(short["missed_detection"])
        self.assertIn("NOT_ASSESSED", short["status"])
        for truth, flag in ((0., 105.), (100., 175.)):
            outside = monitor_evidence({"runtime_cap": truth}, {"runtime_cap": {"observed_at": flag}},
                                       [100.+5*i for i in range(15)])
            self.assertEqual(outside["status"], "NOT_ASSESSED_BOUNDARY_OUTSIDE")
            self.assertIsNone(outside["detected_within_10_ticks"])
            self.assertIsNone(outside["missed_detection"])

    def test_every_certificate_field_is_checked_independently(self):
        records, _ = fixture()
        for key, value in (("lambda_min", -9.), ("lambda_safe", -7), ("dmax", 7),
                           ("m", 9), ("L", 99), ("K", 999), ("bound_clamped", 0.)):
            changed = copy.deepcopy(records)
            changed[1][key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                analyze_run(changed, expected_ticks=2)

    def test_discrete_lag_can_fail_while_wall_startup_and_d7_are_normal(self):
        records, _ = fixture((1,)*10, intervals=[5.]+[4.]*9)
        extra = [dict(type="pod_event", event_type="create", pod="later", at=105.1, accepted=True, command_tick=0),
                 dict(type="pod_event", event_type="ready", pod="later", at=136.9),
                 dict(type="pod_event", event_type="first_service", pod="later", at=137.2, tick=9, discrete_lag=9),
                 dict(type="pod_event", event_type="delete", pod="later", at=141.5)]
        records[-1:-1] = extra
        for row in records:
            if row["type"] == "tick" and row["index"] == 9:
                row["n_start"] = 2
        result = analyze_run(records, expected_ticks=10)
        self.assertEqual(result["independent_truth"], {})
        self.assertEqual(result["max_discrete_lag"], 9)
        self.assertEqual(result["certificate_applicability"], "NOT_OBSERVED_SUPPORTED")
        self.assertTrue(result["loop_run_pass"])

    def test_final_boundary_first_service_and_early_warmup_events_are_valid(self):
        records, _ = fixture()
        records[-1:-1] = [dict(type="pod_event", event_type="create", pod="last", at=105.1, accepted=True, command_tick=0),
            dict(type="pod_event", event_type="ready", pod="last", at=110.1),
            dict(type="pod_event", event_type="first_service", pod="last", at=110.2, tick=2, discrete_lag=2),
            dict(type="pod_event", event_type="delete", pod="last", at=110.5)]
        # A warmup event can be flushed before the sender start/meta arrives.
        initial_create = next(row for row in records if row.get("event_type") == "create")
        records.remove(initial_create)
        records.insert(0, initial_create)
        result = analyze_run(records, expected_ticks=2)
        self.assertEqual(result["max_discrete_lag"], 2)
        self.assertEqual(result["certificate_applicability"], "OBSERVED_CHECKS_SUPPORTED_CONDITIONAL")

    def test_nan_counts_mixed_identity_incomplete_events_and_window_tamper_rejected(self):
        base, requests = fixture()
        mutations = [lambda r: r[1].update(y_count=1.),
                     lambda r: r[1].update(y_count=True),
                     lambda r: r[1].update(boundary_mono=float("nan")),
                     lambda r: r[1].update(arm="HPA-locked"),
                     lambda r: r[1].update(planned_window_count=0),
                     lambda r: r[1].update(latency_window_p95_ms=2.),
                     lambda r: r.pop(-2),  # Delete boundary disappears.
                     lambda r: r[-1].update(written=0)]
        for mutate in mutations:
            records = copy.deepcopy(base)
            mutate(records)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                analyze_run(records, expected_ticks=2, request_rows=requests)
        with self.assertRaises(ValueError):
            analyze_run(base, expected_ticks=2, request_rows=requests[1:])

    def test_original_plan_hash_and_every_offset_required_for_same_plan_evidence(self):
        records, requests = fixture()
        raw = b"request_id,offset_ns\n0,0\n1,5000000000\n"
        records[0]["schedule_sha256"] = hashlib.sha256(raw).hexdigest()
        self.assertEqual(analyze_run(records, expected_ticks=2, request_rows=requests,
            schedule_bytes=raw)["schedule_hash_content_verification"], "VERIFIED")
        changed = copy.deepcopy(requests)
        for key in ("offset_ns", "written_ns", "completed_ns"):
            changed[0][key] = str(int(changed[0][key])+1_000_000)
        # Same tick count and latency cannot hide a changed absolute plan.
        with self.assertRaises(ValueError):
            analyze_run(records, expected_ticks=2, request_rows=changed, schedule_bytes=raw)
        with self.assertRaises(ValueError):
            analyze_run(records, expected_ticks=2, request_rows=requests, schedule_bytes=raw+b"\n")

    def test_adjacent_observed_intervals_and_per_run_95_percent_threshold(self):
        records, _ = fixture((1,)*20, intervals=[4., 6., 6.1]+[5.]*17)
        result = analyze_run(records, expected_ticks=20)
        self.assertEqual(result["loop_in_range_fraction"], .95)
        self.assertTrue(result["loop_run_pass"])
        records, _ = fixture((1,)*20, intervals=[6.1]*2+[5.]*18)
        self.assertFalse(analyze_run(records, expected_ticks=20)["loop_run_pass"])


def matrix():
    rows = []
    for i in range(6):
        for arm in ("HPA-locked", "HPA-wrapped", "HPA-conservative"):
            for delay, repeats in ((5, 1), (15, 1), (30, 2)):
                for repeat in range(1, repeats+1):
                    wrapped = arm == "HPA-wrapped"
                    # Ratio of mean costs=1.2; mean of per-repeat ratios=1.4.
                    cost = (90. if repeat == 1 else 150.) if wrapped else (50. if repeat == 1 else 150.)
                    if delay != 30:
                        cost = 120. if wrapped else 100.
                    rows.append(dict(status="RUN_LOG_ANALYZED", run_id=f"{i}-{arm}-{delay}-{repeat}",
                        source="source", app=str(i), arm=arm, delay=delay, repeat=repeat,
                        schedule_sha256=str(i)*64, paid_replica_seconds=cost,
                        schedule_hash_content_verification="VERIFIED",
                        capacity_violation_rate=.04 if wrapped else .06,
                        loop_interval_count=240, loop_in_range_count=240, loop_run_pass=True))
    return rows


class MatrixAnalysis(unittest.TestCase):
    def test_d5_repeat_means_before_ratios_and_nominal_cost_limit_125(self):
        result = analyze_matrix(matrix())
        self.assertEqual(result["high_improved_fragments"], 6)
        self.assertAlmostEqual(result["high_cost_ratio_median"], 1.2)
        self.assertTrue(result["high_cost_pass"])
        self.assertTrue(result["nominal_pass"])  # 1.2 is allowed, not a 1.0 guard.
        self.assertEqual(result["fault_matrix"], "NOT_ASSESSED")

    def test_ties_do_not_count_incomplete_or_duplicate_slots_cannot_pass(self):
        rows = matrix()
        for row in rows:
            row["capacity_violation_rate"] = .06
        self.assertEqual(analyze_matrix(rows)["high_improved_fragments"], 0)
        self.assertEqual(analyze_matrix(rows[:-1])["status"], "NOT_ASSESSED")
        with self.assertRaises(ValueError):
            analyze_matrix(rows+[rows[0]])
        rows[0]["schedule_sha256"] = "f"*64
        with self.assertRaises(ValueError):
            analyze_matrix(rows)

    def test_repeat_identity_and_unique_run_id_cannot_be_replaced_with_reruns(self):
        for delay, repeat, replacement in ((30, 2, 99), (5, 1, 0)):
            rows = matrix()
            for row in rows:
                if row["delay"] == delay and row["repeat"] == repeat:
                    row["repeat"] = replacement
            with self.assertRaises(ValueError):
                analyze_matrix(rows)
        rows = matrix()
        rows[1]["run_id"] = rows[0]["run_id"]
        with self.assertRaises(ValueError):
            analyze_matrix(rows)

    def test_pooled_timing_99_percent_is_separate_from_every_run_95(self):
        rows = matrix()
        for row in rows:
            row["loop_in_range_count"] = 228
        result = analyze_matrix(rows)
        self.assertTrue(result["timing"]["every_run_pass"])
        self.assertFalse(result["timing"]["pooled_pass"])

    def test_same_hash_label_without_plan_content_cannot_complete_matrix(self):
        rows = matrix()
        rows[0]["schedule_hash_content_verification"] = "NOT_ASSESSED"
        self.assertEqual(analyze_matrix(rows)["status"], "NOT_ASSESSED")


if __name__ == "__main__":
    unittest.main()
