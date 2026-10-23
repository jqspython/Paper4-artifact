"""v3 pure properties with artificial counts/events; no containers or trace IO."""

from decimal import Decimal
import unittest

from sysbench.control import ContractMonitor, TickController, certificate, hpa_request


class SystemControlProperties(unittest.TestCase):
    def monitor(self):
        self.clock_value = 0.
        return ContractMonitor(2., clock=lambda: self.clock_value)

    def test_count_units_tick_start_capacity_and_old_lambda_dispatch(self):
        monitor = self.monitor()
        self.assertEqual(monitor.capacity_count, 10.)
        self.assertEqual(hpa_request(10, 2., 1.), 1)
        self.assertEqual(hpa_request(11, 2., 1.), 2)
        control = TickController(monitor)
        seen = []
        record = control.finish_tick(0, 11, 1, 2., ready_now=4, inflight=0,
                                     emit=lambda target: seen.append((target, control.value)))
        self.assertEqual(seen, [(4, 0.)])
        self.assertEqual((record["request"], record["loss"], record["lambda_t"]), (2, 1, 0.))
        self.assertAlmostEqual(record["lambda_final"], -.475)
        self.assertTrue(record["arithmetic_ok"])
        self.assertEqual(monitor.loss_total, 1)  # Four newly ready pods cannot rewrite n_start=1.

    def test_integer_clip_saturation_is_not_an_emitted_request_fault(self):
        monitor = self.monitor()
        control = TickController(monitor, wrapped=False)
        issued = []
        first = control.finish_tick(0, 0, 1, 100., ready_now=1, inflight=0, emit=issued.append)
        second = control.finish_tick(1, 0, 1, 100., ready_now=1, inflight=2, emit=issued.append)
        self.assertEqual(issued, [3, 5])  # Rate applies to ready+pending, not just ready.
        self.assertEqual((first["request"], second["request"]), (8, 8))
        self.assertTrue(first["saturated"])
        self.assertTrue(first["contract_valid"])
        self.assertEqual(first["certificate_status"], "NOT_APPLICABLE")
        self.assertIsNone(first["lambda_final"])
        rounded = TickController(self.monitor(), wrapped=False).finish_tick(
            0, 0, 1, 1.1, ready_now=1, inflight=0, emit=lambda _: None)
        self.assertEqual(rounded["request"], 2)

    def test_inflight_never_cancelled_and_stable_downscale_has_no_extra_rate_limit(self):
        control = TickController(self.monitor(), wrapped=False)
        for tick in range(5):
            record = control.finish_tick(tick, 0, 1, 1., ready_now=1, inflight=3, emit=lambda _: None)
            self.assertEqual(record["command"], 4)
        record = control.finish_tick(5, 0, 4, 1., ready_now=4, inflight=0, emit=lambda _: None)
        self.assertEqual(record["command"], 1)
        fresh = TickController(self.monitor(), wrapped=False)
        for tick in range(4):
            record = fresh.finish_tick(tick, 0, 8, 1., ready_now=8, inflight=0, emit=lambda _: None)
            self.assertEqual(record["command"], 8)
        record = fresh.finish_tick(4, 0, 8, 1., ready_now=8, inflight=0, emit=lambda _: None)
        self.assertEqual(record["command"], 1)

    def test_stability_is_recent_maximum_not_only_last_recommendation(self):
        control = TickController(self.monitor(), wrapped=False)
        for tick, target in enumerate((8, 1, 1, 1, 1)):
            record = control.finish_tick(tick, 0, 8, target, ready_now=8, inflight=0, emit=lambda _: None)
        self.assertEqual(record["command"], 8)
        record = control.finish_tick(5, 0, 8, 1., ready_now=8, inflight=0, emit=lambda _: None)
        self.assertEqual(record["command"], 1)

    def test_full_create_to_ready_clock_strict_threshold_and_first_flag_latches(self):
        monitor = self.monitor()
        monitor.pod_requested("pod-1")
        self.clock_value = 35.
        monitor.observe_state(runtime_cap=8)
        self.assertFalse(monitor.flags)
        self.clock_value = 35.01
        monitor.observe_state(runtime_cap=8)
        first = dict(monitor.flags["startup_bound"])
        self.clock_value = 40.
        self.assertEqual(monitor.pod_ready("pod-1"), 40.)
        self.assertEqual(monitor.flags["startup_bound"], first)
        self.assertEqual(first["observed_at"], 35.01)
        self.assertFalse(monitor.pending)
        nominal = self.monitor()
        nominal.pod_requested("pod-2")
        self.clock_value = 35.
        self.assertEqual(nominal.pod_ready("pod-2"), 35.)
        self.assertFalse(nominal.flags)

    def test_each_contract_fault_is_passive_for_baselines_and_keeps_observed_time(self):
        monitor = self.monitor()
        monitor.observe_tick(0, 80, 8)
        self.assertFalse(monitor.flags)  # Count=5*c_rps*R is within the envelope.
        self.clock_value = 5.
        monitor.observe_tick(1, 81, 8, runtime_cap=4, external_request=9)
        self.assertEqual(set(monitor.flags), {"demand_envelope", "runtime_cap", "request_above_maximum"})
        self.assertTrue(all(flag["observed_at"] == 5. for flag in monitor.flags.values()))
        self.clock_value = 10.
        monitor.observe_tick(2, 0, 8)
        self.assertTrue(all(flag["observed_at"] == 5. for flag in monitor.flags.values()))
        snapshot = monitor.snapshot()
        self.assertFalse(snapshot["contract_valid"])
        self.assertEqual(snapshot["certificate_status"], "NOT_APPLICABLE")

    def test_monotonic_clock_and_consecutive_tick_guards(self):
        monitor = self.monitor()
        monitor.observe_tick(0, 0, 1, when=5.)
        for index, when in ((1, 4.), (0, 5.), (2, 10.)):
            with self.assertRaises(ValueError):
                monitor.observe_tick(index, 0, 1, when=when)
        self.assertEqual(monitor.ticks, 1)
        with self.assertRaises(ValueError):
            monitor.pod_ready("unrecorded", when=5.)
        monitor.pod_requested("created", when=5.)
        with self.assertRaises(ValueError):
            monitor.pod_requested("created", when=5.)

    def test_unprojected_update_certificate_offline_decimal_and_revised_bound(self):
        monitor = self.monitor()
        control = TickController(monitor)
        offline, minimum, losses = Decimal(0), Decimal(0), 0
        for tick in range(240):
            loss = int(tick % 4 == 0)
            self.clock_value = 5. * (tick + 1)
            record = control.finish_tick(tick, 11 if loss else 0, 1, 1., ready_now=1,
                                         inflight=0, emit=lambda _: None)
            losses += loss
            offline += Decimal('.5') * (Decimal('.05') - loss)
            minimum = min(minimum, offline)
            self.assertLessEqual(abs(record["lambda_final"] - float(offline)), 1e-9)
            self.assertLessEqual(abs(record["lambda_min"] - float(minimum)), 1e-9)
            self.assertLessEqual(abs(record["identity_residual"]), 1e-9)
        self.assertEqual((record["dmax"], record["m"], record["L"], record["K"]), (8, 4, 11, 220))
        self.assertAlmostEqual(record["bound_raw"], 31/30)
        self.assertEqual(record["bound_clamped"], 1.)
        self.assertEqual(record["loss_total"], losses)
        self.assertLess(record["lambda_final"], 0.)
        short = certificate(100, 0, lambda_final=2.5, lambda_min=0.)
        self.assertAlmostEqual(short["bound_raw"], 2.41)
        self.assertEqual(short["bound_clamped"], 1.)
        original = certificate(240, 0, lambda_final=6., lambda_min=0., startup_bound_seconds=30)
        self.assertAlmostEqual(original["bound_raw"], .95)  # Historical arithmetic, not the current contract.

    def test_command_error_does_not_erase_loss_or_feedback(self):
        control = TickController(self.monitor())
        def failed_emit(_):
            self.assertEqual(control.value, 0.)
            raise RuntimeError("artificial dispatch failure")
        record = control.finish_tick(0, 11, 1, 2., ready_now=1, inflight=0, emit=failed_emit)
        self.assertFalse(record["issued"])
        self.assertEqual(record["loss_total"], 1)
        self.assertAlmostEqual(record["lambda_final"], -.475)
        self.assertTrue(record["arithmetic_ok"])
        self.assertEqual(record["certificate_status"], "OUTSIDE_OBSERVED_CONTRACT")
        self.assertIn("command_error", record["first_flags"])


class OfflineAnalysisPath(unittest.TestCase):
    def test_roundtrip_missing_measurements_and_tamper_guard(self):
        from sysbench.analyze import analyze_rows
        from sysbench.feasibility import summary
        control = TickController(ContractMonitor(10))
        rows = [control.finish_tick(i, 60 if i % 4 == 0 else 1, 1, 2,
                ready_now=1, inflight=0, emit=lambda _: None, when=5*(i+1)) for i in range(240)]
        result = analyze_rows(rows, 10)
        self.assertLessEqual(result["max_certificate_error"], 1e-9)
        self.assertEqual(result["paid_replica_seconds"], "NOT_MEASURED")
        rows[0]["y_count"] = 1
        with self.assertRaises(ValueError):
            analyze_rows(rows, 10)
        payload = {"metrics": {}}
        self.assertEqual(summary('PAPER4_SUMMARY={"metrics":{}}'), payload)
        self.assertEqual(summary('time="x" msg="PAPER4_SUMMARY={\\"metrics\\":{}}" source=console'), payload)


if __name__ == "__main__":
    unittest.main()
