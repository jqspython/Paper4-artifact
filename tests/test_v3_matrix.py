"""ID-only manifest and D6 state-machine checks; no traces or actual launches."""
from contextlib import redirect_stdout
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from sysbench import matrix


PROJECT = Path(__file__).resolve().parents[1]


class MatrixLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".v3-matrix-test-", dir=PROJECT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root/"configs").mkdir()
        (self.root/"results").mkdir()
        self.ids = matrix.read_csv(PROJECT/"results/v3-sys-fragment-ids.csv")
        with (self.root/"results/ids.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.ids[0])
            writer.writeheader()
            writer.writerows(self.ids)
        self.write_config()
        self.manifest = matrix.write_manifest(self.root)
        self.allow_freeze = patch("sysbench.matrix.require_frozen", return_value="1"*40)
        self.allow_freeze.start()
        self.addCleanup(self.allow_freeze.stop)
        # State-machine tests inject a receipt validator; one test below uses
        # complete artificial JSONL/request/plan artifacts through the analyzer.
        self.allow_artifact = patch("sysbench.matrix.verified_artifact", return_value={})
        self.allow_artifact.start()
        self.addCleanup(self.allow_artifact.stop)

    def write_config(self, *, status="FROZEN", high=30):
        # Artificial metadata is not an approval or frozen real experiment.
        text = f'''status = "{status}"
freeze_tag = "test-freeze"
[selection]
ids = "results/ids.csv"
seed = 20261022
[controller]
ticks = 240
[matrix]
arms = ["HPA-locked", "HPA-wrapped", "HPA-conservative"]
repeats = [1, 1, 2]
injection_seconds = [5, 15, {high}]
formal_runs = 72
fault_runs = 6
[faults]
start_tick = 60
end_tick_exclusive = 120
fragments = ["huawei_public2023/1455", "huawei_private2023/149"]
'''
        (self.root/"configs/v3-sys.toml").write_text(text, encoding="utf-8")

    def reserve(self, slot=None, **kwargs):
        return matrix.reserve(self.root, slot or self.manifest[0]["slot_id"], **kwargs)

    def close(self, start, *, valid=True, inside=240, total=240, **kwargs):
        return matrix.finish(self.root, start["slot_id"], int(start["attempt"]), run_id=start["run_id"],
            infrastructure_ok=valid, in_range_ticks=inside, total_ticks=total,
            log_path=f"work/v3-sys/{start['run_id']}.jsonl" if valid else "", **kwargs)

    def test_exact_72_plus_6_hash_selection_shared_plan_seed_and_fixed_shuffle(self):
        config, rebuilt = matrix.config_and_plan(self.root)
        self.assertEqual(rebuilt, self.manifest)
        self.assertEqual(sum(row["kind"] == "normal" for row in rebuilt), 72)
        self.assertEqual(sum(row["kind"] == "fault" for row in rebuilt), 6)
        formal = sorted((row for row in self.ids if row["role"] == "formal-proposed"),
            key=lambda r: hashlib.sha256(f"v3-sys|20261022|{r['source']}|{r['app']}".encode()).digest())
        fault_pairs = {(r["source"], r["app"]) for r in rebuilt if r["kind"] == "fault"}
        self.assertEqual(fault_pairs, {(r["source"], r["app"]) for r in formal[:2]})
        for identity in formal:
            rows = [r for r in rebuilt if (r["source"], r["app"]) == (identity["source"], identity["app"])]
            ordinary = [r for r in rows if r["arrival_fault"] == "none"]
            raw = hashlib.sha256(f"v3-sys|20261022|{identity['source']}|{identity['app']}".encode()).digest()
            self.assertEqual({r["seed"] for r in rows}, {str(int.from_bytes(raw[:8], "big"))})
            self.assertEqual(len({r["plan_key"] for r in ordinary}), 1)
            if any(r["fault"] == "demand3" for r in rows):
                self.assertEqual(len({r["plan_key"] for r in rows}), 2)
        self.assertEqual([int(row["order"]) for row in rebuilt], list(range(1, 79)))
        self.assertNotEqual([r["slot_id"] for r in rebuilt], sorted(r["slot_id"] for r in rebuilt))

    def test_d9_only_replaces_high_delay_and_never_changes_plans_or_slot_count(self):
        self.write_config(high=25)
        _, changed = matrix.config_and_plan(self.root)
        before = {r["slot_id"]: r for r in self.manifest}
        self.assertEqual(len(changed), 78)
        for row in changed:
            old = before[row["slot_id"]]
            self.assertEqual(row["plan_key"], old["plan_key"])
            self.assertEqual(row["seed"], old["seed"])
            self.assertEqual(row["delay"], "25" if old["delay"] == "30" else old["delay"])
        with self.assertRaises(ValueError):
            matrix.write_manifest(self.root)  # Never silently overwrite the old plan.

    def test_draft_cli_plan_is_metadata_only_but_reserve_remains_blocked(self):
        self.write_config(status="NOT_FROZEN")
        self.allow_freeze.stop()
        output = io.StringIO()
        with redirect_stdout(output):
            rows = matrix.main(["--stage", "plan"], root=self.root)
        self.assertEqual(len(rows), 78)
        self.assertIn("no demand access", output.getvalue())
        with self.assertRaises(RuntimeError):
            self.reserve()
        self.assertFalse((self.root/matrix.LEDGER).exists())

    def test_append_only_invalid_then_first_valid_is_final_and_unique(self):
        start = self.reserve()
        path = self.root/matrix.LEDGER
        first = path.read_bytes()
        self.close(start, inside=227)
        self.assertTrue(path.read_bytes().startswith(first))
        retry = self.reserve()
        self.assertEqual(retry["attempt"], "1")
        second = path.read_bytes()
        self.assertEqual(self.close(retry, inside=228)[0]["event"], "VALID")
        self.assertTrue(path.read_bytes().startswith(second))
        eligible = matrix.eligible_attempts(self.root)
        self.assertEqual([(r["slot_id"], r["attempt"]) for r in eligible], [(start["slot_id"], "1")])
        # Pooled <99% is not authority to reopen this individually valid slot.
        with self.assertRaises(RuntimeError):
            self.reserve()
        with self.assertRaises(ValueError):
            self.close(retry)

    def test_initial_plus_two_reruns_exhausted_without_a_fourth_attempt(self):
        for attempt in range(3):
            start = self.reserve()
            self.assertEqual(start["attempt"], str(attempt))
            events = self.close(start, valid=False, inside=None, total=None)
        self.assertEqual([r["event"] for r in events], ["INVALID", "EXHAUSTED"])
        rows = matrix.read_csv(self.root/matrix.LEDGER, matrix.EVENT_FIELDS)
        self.assertEqual([r["event"] for r in rows], ["STARTED", "INVALID"]*3+["EXHAUSTED"])
        self.assertEqual(matrix.eligible_attempts(self.root), [])
        with self.assertRaises(RuntimeError):
            self.reserve()

    def test_active_slot_wrong_attempt_run_identity_and_log_cannot_be_replaced(self):
        start = self.reserve()
        original = (self.root/matrix.LEDGER).read_bytes()
        with self.assertRaises(RuntimeError):
            self.reserve()
        for slot, attempt, run_id in (("normal-002", 0, start["run_id"]),
                                      (start["slot_id"], 1, start["run_id"]),
                                      (start["slot_id"], 0, "another-run")):
            with self.assertRaises(ValueError):
                matrix.finish(self.root, slot, attempt, run_id=run_id, infrastructure_ok=False)
        with self.assertRaises(ValueError):
            matrix.finish(self.root, start["slot_id"], 0, run_id=start["run_id"], infrastructure_ok=True,
                in_range_ticks=240, total_ticks=240, log_path="work/another-run.jsonl")
        self.assertEqual((self.root/matrix.LEDGER).read_bytes(), original)

    def test_manifest_source_and_ledger_arm_tamper_are_rejected(self):
        start = self.reserve()
        rows = matrix.read_csv(self.root/matrix.LEDGER, matrix.EVENT_FIELDS)
        rows[0]["arm"] = "HPA-conservative" if rows[0]["arm"] != "HPA-conservative" else "HPA-locked"
        with (self.root/matrix.LEDGER).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=matrix.EVENT_FIELDS)
            writer.writeheader(); writer.writerows(rows)
        with self.assertRaises(ValueError):
            matrix.eligible_attempts(self.root)
        manifest = copy.deepcopy(self.manifest)
        manifest[0]["source"] = "replacement"
        with (self.root/matrix.MANIFEST).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=matrix.FIELDS)
            writer.writeheader(); writer.writerows(manifest)
        with self.assertRaises(ValueError):
            self.reserve("normal-002")

    def test_incomplete_timing_invalid_and_failed_infrastructure_keeps_good_timing_invalid(self):
        for kwargs in (dict(total=239, inside=239), dict(valid=False), dict(total=None, inside=None)):
            start = self.reserve()
            self.assertEqual(self.close(start, **kwargs)[0]["event"], "INVALID")
        self.assertEqual(matrix.eligible_attempts(self.root), [])

    def test_finish_logs_failure_even_when_freeze_check_would_now_fail(self):
        start = self.reserve()
        self.allow_freeze.stop()
        self.write_config(status="NOT_FROZEN")
        self.assertEqual(self.close(start, valid=False)[0]["event"], "INVALID")

    def test_changed_freeze_can_only_finish_invalid_even_with_good_timing(self):
        start = self.reserve()
        self.allow_freeze.stop()
        self.write_config(status="NOT_FROZEN")
        event = self.close(start)[0]
        self.assertEqual(event["event"], "INVALID")
        self.assertIn("FREEZE_OR_ARTIFACT_INVALID", event["reason"])
        self.assertEqual(matrix.eligible_attempts(self.root), [])

    def test_new_ledger_path_cannot_reset_terminal_or_attempt_count(self):
        start = self.reserve()
        self.close(start)
        with self.assertRaises(ValueError):
            matrix.reserve(self.root, start["slot_id"], ledger_path="results/reset.csv")
        with self.assertRaises(ValueError):
            matrix.eligible_attempts(self.root, ledger_path="results/reset.csv")
        self.assertFalse((self.root/"results/reset.csv").exists())

    def test_fixed_order_and_single_active_attempt_are_enforced(self):
        later = self.manifest[1]["slot_id"]
        with self.assertRaises(RuntimeError):
            self.reserve(later)
        first = self.reserve()
        with self.assertRaises(RuntimeError):
            self.reserve(later)
        self.close(first)
        second = self.reserve(later)
        self.close(second)
        self.assertEqual([r["slot_id"] for r in matrix.eligible_attempts(self.root)],
                         [first["slot_id"], second["slot_id"]])

    def test_partial_failure_artifact_is_kept_and_complete_receipt_is_independently_bound(self):
        self.allow_artifact.stop()
        first = self.reserve()
        partial = f"work/{first['run_id']}.partial.jsonl"
        events = matrix.finish(self.root, first["slot_id"], 0, run_id=first["run_id"],
            infrastructure_ok=False, log_path=partial)
        self.assertEqual(events[0]["log_path"], partial)
        retry = self.reserve()
        self.materialize(retry)
        self.assertEqual(self.close(retry)[0]["event"], "VALID")

    def test_existing_complete_file_with_wrong_source_is_not_eligible(self):
        self.allow_artifact.stop()
        start = self.reserve()
        path = self.materialize(start)
        records = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        records[0]["source"] = "another-source"
        path.write_text("\n".join(json.dumps(r) for r in records)+"\n", encoding="utf-8")
        event = self.close(start)[0]
        self.assertEqual(event["event"], "INVALID")
        self.assertIn("source", event["reason"])

    def materialize(self, start):
        from tests.test_v3_analysis import fixture
        records, requests = fixture((1,)*240, arm=start["arm"])
        folder = self.root/"work/v3-sys"
        folder.mkdir(parents=True, exist_ok=True)
        plan = b"request_id,offset_ns\n"+b"".join(f"{r['request_id']},{r['offset_ns']}\n".encode() for r in requests)
        (folder/"plan.csv").write_bytes(plan)
        request_name = start["run_id"]+"-requests.csv"
        with (folder/request_name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=requests[0])
            writer.writeheader(); writer.writerows(requests)
        records[0].update(run_id=start["run_id"], source=start["source"], app=start["app"],
            delay=int(start["delay"]), repeat=int(start["repeat"]), fault=start["fault"], ticks=240,
            schedule_sha256=hashlib.sha256(plan).hexdigest(), schedule_csv="plan.csv", request_csv=request_name)
        path = folder/(start["run_id"]+".jsonl")
        path.write_text("\n".join(json.dumps(r, allow_nan=False) for r in records)+"\n", encoding="utf-8")
        return path

    def test_concurrent_writer_lock_is_not_bypassed_and_leaves_no_attempt(self):
        lock = (self.root/matrix.LEDGER).with_suffix(".csv.lock")
        lock.touch()
        with self.assertRaises(RuntimeError):
            self.reserve()
        self.assertFalse((self.root/matrix.LEDGER).exists())
        lock.unlink()

    def test_cli_reserve_finish_use_the_same_state_machine(self):
        output = io.StringIO()
        with redirect_stdout(output):
            start = matrix.main(["--stage", "reserve", "--slot", self.manifest[0]["slot_id"]], root=self.root)
            matrix.main(["--stage", "finish", "--slot", start["slot_id"], "--attempt", "0",
                "--run-id", start["run_id"], "--infrastructure", "ok", "--in-range-ticks", "240",
                "--total-ticks", "240", "--log", f"work/{start['run_id']}.jsonl"], root=self.root)
        self.assertIn("STARTED", output.getvalue())
        self.assertIn("VALID", output.getvalue())
        self.assertEqual(len(matrix.eligible_attempts(self.root)), 1)


class FreezeChecks(unittest.TestCase):
    def test_tag_diff_and_untracked_source_all_gate_formal_reserve(self):
        config = dict(status="FROZEN", freeze_tag="test-freeze")
        ok = subprocess.CompletedProcess([], 0, "", "")
        commit = subprocess.CompletedProcess([], 0, "a"*40+"\n", "")
        with patch("sysbench.matrix.subprocess.run", side_effect=[commit, ok, ok]) as call:
            self.assertEqual(matrix.require_frozen(PROJECT, config), "a"*40)
            self.assertIn("refs/tags/test-freeze^{commit}", call.call_args_list[0].args[0])
            self.assertEqual(tuple(call.call_args_list[1].args[0][-len(matrix.CRITICAL):]), matrix.CRITICAL)
        for responses in ((subprocess.CompletedProcess([], 1, "", "missing"),),
                          (commit, subprocess.CompletedProcess([], 1, "", "dirty"), ok),
                          (commit, ok, subprocess.CompletedProcess([], 0, "sysbench/untracked.py\n", ""))):
            with patch("sysbench.matrix.subprocess.run", side_effect=responses), self.assertRaises(RuntimeError):
                matrix.require_frozen(PROJECT, config)


if __name__ == "__main__":
    unittest.main()
