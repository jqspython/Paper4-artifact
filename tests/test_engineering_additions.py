import csv
from datetime import datetime
from pathlib import Path
import tempfile
import unittest

from paper4_monitor_matrix import matrix_rows, matrix_status
from paper4_public_reproduce import reproduce
from sysbench.contract_assumption import SHANGHAI, _time_gate, assumption_status, design_rows, observed_steps


class EngineeringAdditionsTests(unittest.TestCase):
    def test_kind_contract_plan_is_exact_and_offline(self):
        rows = design_rows()
        self.assertEqual(len(rows), 6)
        self.assertEqual(sum(row["new_pods"] for row in rows), 24)
        self.assertEqual({(row["delay_seconds"], row["model_d"]) for row in rows}, {(120, 3), (300, 6)})
        self.assertEqual(observed_steps(120.01), 3)
        self.assertEqual(observed_steps(300.01), 6)
        with self.assertRaises(RuntimeError):
            _time_gate(datetime(2026, 10, 27, 23, 59, tzinfo=SHANGHAI))
        _time_gate(datetime(2026, 10, 28, tzinfo=SHANGHAI))
        with self.assertRaises(RuntimeError):
            _time_gate(datetime(2026, 10, 29, 12, tzinfo=SHANGHAI))
        censored = [dict(within_model_d="")]
        self.assertEqual(assumption_status(censored, []), "INCOMPLETE_AFTER_TIMING")
        pods = [dict(within_model_d="1") for _ in range(24)]
        downs = [dict(blocked="1") for _ in range(12)]
        self.assertEqual(assumption_status(pods, downs), "ASSUMPTIONS_HELD_IN_THIS_CHECK")
        pods[0]["within_model_d"] = "0"
        self.assertEqual(assumption_status(pods, downs), "ASSUMPTIONS_NOT_ALWAYS_HELD")

    def test_monitor_matrix_has_zero_false_positives_and_complete_fault_detection(self):
        rows = matrix_rows()
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["detected"] == 9 for row in rows[:3]))
        self.assertTrue(all(row["min_delay_ticks"] == row["max_delay_ticks"] == 0 for row in rows[:3]))
        self.assertEqual(rows[-1]["trajectories"], 1000)
        self.assertEqual(rows[-1]["false_positive_trajectories"], 0)
        self.assertEqual(rows[-1]["status"], "PASS")
        self.assertEqual(matrix_status(rows), "PASS")
        rows[0] = dict(rows[0], detected=0, status="DETECTION_MISSING")
        self.assertEqual(matrix_status(rows), "FAIL")

    def test_public_reproduction_reports_insufficient_statistics_not_fake_values(self):
        root = Path(__file__).resolve().parents[1]
        manuscript = root / "manuscript/submission.md"
        if not manuscript.is_file():
            self.skipTest("Unpublished manuscript source is intentionally not bundled")
        with tempfile.TemporaryDirectory() as directory:
            public = Path(directory) / "results"; public.mkdir()
            names = ("v2-lite-confirm-endpoints.csv", "v2-lite-confirm-risk.csv",
                     "v2-lite-confirm-descriptive.csv", "v1.1-static-reference-summary.csv",
                     "v1.1-lambda-diagnostic.csv", "v1.1-azure2021-qualification.csv",
                     "v3-sys-closure.csv")
            for name in names:
                (public / name).write_bytes((root / "results" / name).read_bytes())
            output = Path(directory) / "output"
            status, comparisons = reproduce(public, output, manuscript)
            self.assertEqual(status, "INPUT_INCOMPLETE")
            self.assertFalse(any(row["status"] == "DIFF" for row in comparisons))
            self.assertTrue(any(row["artifact"] == "Table 7" and row["status"] == "INPUT_INCOMPLETE"
                                for row in comparisons))
            self.assertTrue(any(row["artifact"] == "Figure 1" and row["status"] == "INPUT_INCOMPLETE"
                                for row in comparisons))
            with (output / "comparison.csv").open(encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), len(comparisons))


if __name__ == "__main__":
    unittest.main()
