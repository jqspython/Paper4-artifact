"""ID-only manifest and D6 state-machine checks; no traces or actual launches."""
from contextlib import redirect_stdout, nullcontext
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

    def write_config(self, *, status="FROZEN", high=30, capacity=96):
        # Artificial metadata is not an approval or frozen real experiment.
        text = f'''status = "{status}"
freeze_tag = "test-freeze"
[calibration]
capacity_rps = {capacity}
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

    def preclock_revision(self, *, measured=False):
        start=self.reserve()
        path=self.root/'work'/'v3-sys'/(start['run_id']+'.partial.jsonl'); path.parent.mkdir(parents=True)
        records=[dict(type='infrastructure_failure',error='synthetic pre-clock refusal')]
        if measured: records.append(dict(type='tick'))
        path.write_text('\n'.join(json.dumps(r) for r in records)+'\n',encoding='utf-8')
        stderr=self.root/'work'/'v3-sys'/'host'/(start['run_id']+'.stderr.log')
        stderr.parent.mkdir(); stderr.write_text('synthetic independently reviewed pre-clock refusal',encoding='utf-8')
        matrix.finish(self.root,start['slot_id'],0,run_id=start['run_id'],infrastructure_ok=False,
                      log_path=path.relative_to(self.root).as_posix())
        config=self.root/'configs'/'v3-sys.toml'
        config.write_text(config.read_text('utf-8').replace('freeze_tag = "test-freeze"','freeze_tag = "test-freeze2"')+
            '\n[execution_compatibility]\nprior_invalid_before_clock_refs = ["test-freeze"]\n'
            '[execution_compatibility.invalid_before_clock_receipts]\n"'+start['run_id']+'" = "'+hashlib.sha256(path.read_bytes()).hexdigest()+'"\n'+
            '[execution_compatibility.invalid_before_clock_stderr_receipts]\n"'+start['run_id']+'" = "'+hashlib.sha256(stderr.read_bytes()).hexdigest()+'"\n',encoding='utf-8')
        return start,path

    def test_declared_preclock_revision_retains_old_bytes_and_spends_original_reruns(self):
        start,path=self.preclock_revision(); original=(self.root/matrix.LEDGER).read_bytes()
        with patch('sysbench.matrix.require_frozen',return_value='2'*40),\
                patch('sysbench.matrix.subprocess.check_output',return_value='1'*40+'\n'):
            retry=self.reserve(); self.assertEqual(retry['attempt'],'1')
            self.assertEqual(retry['freeze_tag'],'test-freeze2'); self.assertEqual(retry['freeze_commit'],'2'*40)
            self.assertTrue((self.root/matrix.LEDGER).read_bytes().startswith(original))
            self.close(retry,valid=False)
            final=self.reserve(); self.assertEqual(final['attempt'],'2')
            self.assertEqual(self.close(final,valid=False)[-1]['event'],'EXHAUSTED')
            with self.assertRaises(RuntimeError): self.reserve()

    def test_preclock_revision_rejects_changed_receipt_or_unknown_parent(self):
        start,path=self.preclock_revision()
        original=path.read_bytes()
        with patch('sysbench.matrix.require_frozen',return_value='2'*40),\
                patch('sysbench.matrix.subprocess.check_output',return_value='3'*40+'\n'):
            with self.assertRaises(ValueError): self.reserve()
        path.write_text(path.read_text('utf-8')+'\n',encoding='utf-8')
        with patch('sysbench.matrix.require_frozen',return_value='2'*40),\
                patch('sysbench.matrix.subprocess.check_output',return_value='1'*40+'\n'):
            with self.assertRaises(ValueError): self.reserve()
        path.write_bytes(original)
        stderr=self.root/'work'/'v3-sys'/'host'/(start['run_id']+'.stderr.log')
        stderr.write_bytes(stderr.read_bytes()+b'changed')
        with patch('sysbench.matrix.require_frozen',return_value='2'*40),\
                patch('sysbench.matrix.subprocess.check_output',return_value='1'*40+'\n'):
            with self.assertRaises(ValueError): self.reserve()

    def test_preclock_revision_cannot_carry_clocked_failure(self):
        self.preclock_revision(measured=True)
        with patch('sysbench.matrix.require_frozen',return_value='2'*40),\
                patch('sysbench.matrix.subprocess.check_output',return_value='1'*40+'\n'):
            with self.assertRaises(ValueError): self.reserve()

    def test_preclock_revision_cannot_reopen_or_inherit_an_eligible_old_version(self):
        start=self.reserve(); self.close(start)
        config=self.root/'configs'/'v3-sys.toml'
        config.write_text(config.read_text('utf-8').replace('freeze_tag = "test-freeze"','freeze_tag = "test-freeze2"')+
            '\n[execution_compatibility]\nprior_invalid_before_clock_refs = ["test-freeze"]\n',encoding='utf-8')
        with patch('sysbench.matrix.require_frozen',return_value='2'*40),\
                patch('sysbench.matrix.subprocess.check_output',return_value='1'*40+'\n'):
            with self.assertRaises(ValueError): self.reserve(self.manifest[1]['slot_id'])

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

    def test_internally_consistent_old_capacity_log_is_not_eligible(self):
        self.allow_artifact.stop()
        self.write_config(capacity=60)
        start = self.reserve()
        self.materialize(start)  # Complete synthetic log was constructed with c=96.
        event = self.close(start)[0]
        self.assertEqual(event["event"], "INVALID")
        self.assertIn("capacity", event["reason"])

    def materialize(self, start):
        from tests.test_v3_analysis import fixture
        records, requests = fixture((1,)*240, arm=start["arm"])
        folder = self.root/"work/v3-sys"
        folder.mkdir(parents=True, exist_ok=True)
        plan = b"request_id,offset_ns\n"+b"".join(f"{r['request_id']},{r['offset_ns']}\n".encode() for r in requests)
        from sysbench.fragments import _plan_metadata
        plan_name=start['plan_key']+'-'+matrix.GENERATOR_VERSION+'-plan.csv'
        (folder/plan_name).write_bytes(plan)
        plan_meta=dict(_plan_metadata(start['source'],start['app'],fault=start['arrival_fault'],capacity=96),
                       plan_key=start['plan_key'],planned_requests=len(requests),schedule_sha256=hashlib.sha256(plan).hexdigest())
        (folder/plan_name).with_suffix('.meta.json').write_text(json.dumps(plan_meta),encoding='utf-8')
        request_name = start["run_id"]+"-requests.csv"
        with (folder/request_name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=requests[0])
            writer.writeheader(); writer.writerows(requests)
        records[0].update(run_id=start["run_id"], source=start["source"], app=start["app"],
            delay=int(start["delay"]), repeat=int(start["repeat"]), fault=start["fault"], ticks=240,
            schedule_sha256=hashlib.sha256(plan).hexdigest(), schedule_csv=plan_name, request_csv=request_name)
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

    def stopped_amendment_fixture(self):
        """Construct the eleven old events and receipts in a temporary test root."""
        self.assertEqual([r['slot_id'] for r in self.manifest[:2]],['normal-021','fault-002'])
        first=self.reserve(); self.close(first,valid=False)
        old=self.reserve(); self.materialize(old); self.close(old)
        for _ in range(3):
            start=self.reserve('fault-002'); self.close(start,valid=False,inside=None,total=None)
            failure=self.root/'work/v3-sys/host'/(start['run_id']+'.failure.log')
            failure.parent.mkdir(parents=True,exist_ok=True); failure.write_bytes(b'synthetic deterministic error')
        folder=self.root/'work/v3-sys'
        for suffix in ('-plan.csv','-plan.meta.json'):
            (folder/(old['plan_key']+suffix)).write_bytes(b'synthetic original plan receipt')
        path=self.root/matrix.LEDGER; original=path.read_bytes()
        receipts={p.relative_to(self.root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in folder.rglob('*') if p.is_file()}
        amendment=dict(id='d6-float64-20261025',authorization='USER_APPROVED_2026-10-25',
            activation_tag='test-freeze2',void_slot='normal-021',void_attempt=1,
            extra_slot='fault-002',extra_attempt=3,legacy_rows=11,
            legacy_ledger_sha256=hashlib.sha256(original).hexdigest(),evidence=receipts)
        config=self.root/'configs/v3-sys.toml'
        text=config.read_text('utf-8').replace('freeze_tag = "test-freeze"','freeze_tag = "test-freeze2"')
        text+='\n[schedule_generator]\nversion = "'+matrix.GENERATOR_VERSION+'"\n[execution_amendment]\n'
        text+='\n'.join(k+' = '+json.dumps(v) for k,v in amendment.items() if k!='evidence')
        text+='\n[execution_amendment.evidence]\n'+'\n'.join(json.dumps(k)+' = '+json.dumps(v) for k,v in receipts.items())+'\n'
        config.write_text(text,encoding='utf-8')
        freeze=patch('sysbench.matrix.require_frozen',return_value='2'*40); freeze.start(); self.addCleanup(freeze.stop)
        frozen_context=text
        refs=patch('sysbench.matrix.subprocess.check_output',side_effect=lambda args,**kw:
            frozen_context if args[1]=='show' else ('2' if 'test-freeze2' in args[-1] else '1')*40+'\n')
        refs.start(); self.addCleanup(refs.stop)
        return amendment,original,old

    def test_approved_amendment_appends_two_bound_events_once_and_preserves_old_bytes(self):
        amendment,original,old=self.stopped_amendment_fixture()
        with self.assertRaises(RuntimeError): self.reserve()  # Approval alone does not clear EXHAUSTED.
        result=matrix.apply_amendment(self.root)
        self.assertEqual(result['status'],'APPLIED_TWO_APPEND_ONLY_EVENTS')
        path=self.root/matrix.LEDGER
        self.assertTrue(path.read_bytes().startswith(original))
        rows=matrix.read_csv(path,matrix.EVENT_FIELDS)
        self.assertEqual([r['event'] for r in rows[-2:]],['VOID_IMPLEMENTATION','RETRY_AUTHORIZED'])
        self.assertEqual(rows[-2]['run_id'],old['run_id'])
        self.assertEqual(rows[-2]['freeze_commit'],'1'*40)  # Original identity never rewritten.
        self.assertTrue(all(r['reason']==matrix.amendment_reason(amendment,'2'*40) for r in rows[-2:]))
        before=path.read_bytes()
        self.assertEqual(matrix.apply_amendment(self.root)['status'],'ALREADY_APPLIED_NO_NEW_EVENTS')
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(matrix.eligible_attempts(self.root),[])  # Old VALID no longer eligible.

    def test_normal_replacement_spends_only_original_a2_and_cannot_reopen_a3(self):
        amendment,_,_=self.stopped_amendment_fixture(); matrix.apply_amendment(self.root)
        start=self.reserve(); self.assertEqual(start['attempt'],'2')
        self.assertEqual(self.close(start,valid=False)[-1]['event'],'EXHAUSTED')
        with self.assertRaises(RuntimeError): self.reserve()
        with self.assertRaises(RuntimeError): self.reserve('fault-002')

    def test_only_fault002_gets_a3_then_exhausts_no_a4_or_other_budget_extension(self):
        amendment,_,_=self.stopped_amendment_fixture(); matrix.apply_amendment(self.root)
        first=self.reserve(); self.materialize(first); self.close(first)
        self.assertEqual([(r['slot_id'],r['attempt']) for r in matrix.eligible_attempts(self.root)],[('normal-021','2')])
        extra=self.reserve('fault-002'); self.assertEqual(extra['attempt'],'3')
        events=self.close(extra,valid=False)
        self.assertEqual([r['event'] for r in events],['INVALID','EXHAUSTED'])
        with self.assertRaises(RuntimeError): self.reserve('fault-002')
        self.assertEqual({r['slot_id'] for r in self.manifest if matrix.retry_limit(r['slot_id'],amendment)==3},{'fault-002'})

    def test_amendment_rejects_changed_old_ledger_artifact_scope_or_reference(self):
        amendment,original,_=self.stopped_amendment_fixture()
        config=matrix.config_and_plan(self.root)[0]
        path=self.root/matrix.LEDGER
        path.write_bytes(original.replace(b'RESERVED_BEFORE_EXECUTION',b'ALTERED_BEFORE_EXECUTION',1))
        with self.assertRaisesRegex(ValueError,'prefix changed'): matrix.apply_amendment(self.root)
        path.write_bytes(original)
        receipt=self.root/next(iter(amendment['evidence']))
        saved=receipt.read_bytes(); receipt.write_bytes(saved+b'changed')
        with self.assertRaisesRegex(ValueError,'artifact changed'): matrix.apply_amendment(self.root)
        receipt.write_bytes(saved)
        with patch('sysbench.matrix.subprocess.check_output',return_value='3'*40+'\n'):
            with self.assertRaisesRegex(ValueError,'identity changed'): matrix.apply_amendment(self.root)
        for key,value in (('extra_slot','fault-003'),('extra_attempt',4),('void_attempt',0),('legacy_rows',10),('authorization','AI_APPROVED')):
            bad=copy.deepcopy(config); bad['execution_amendment'][key]=value
            with self.assertRaises(ValueError): matrix.approved_amendment(bad)
        self.assertEqual(path.read_bytes(),original)

    def test_half_applied_duplicate_or_unbound_administrative_events_are_rejected(self):
        _,_,_=self.stopped_amendment_fixture(); matrix.apply_amendment(self.root)
        path=self.root/matrix.LEDGER; saved=path.read_bytes(); lines=saved.splitlines(keepends=True)
        for changed in (b''.join(lines[:-1]),saved+lines[-1],saved.replace(b'USER_APPROVED_D6:',b'UNBOUND_D6:')):
            path.write_bytes(changed)
            with self.assertRaises(ValueError): matrix.apply_amendment(self.root)
        path.write_bytes(saved)

    def test_missing_generator_or_legacy_valid_is_rejected_by_current_eligibility(self):
        _,_,_=self.stopped_amendment_fixture(); matrix.apply_amendment(self.root)
        current=self.reserve(); path=self.materialize(current); self.close(current)
        records=[json.loads(line) for line in path.read_text('utf-8').splitlines()]
        records[0].pop('generator_version')
        path.write_text('\n'.join(map(json.dumps,records))+'\n',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'legacy VALID'): matrix.eligible_attempts(self.root)

    def test_amended_active_attempt_still_logs_invalid_when_live_config_is_broken(self):
        self.stopped_amendment_fixture(); matrix.apply_amendment(self.root)
        current=self.reserve(); self.materialize(current)
        (self.root/'configs/v3-sys.toml').write_bytes(b'broken TOML')
        events=self.close(current)
        self.assertEqual([r['event'] for r in events],['INVALID','EXHAUSTED'])
        self.assertIn('FREEZE_OR_ARTIFACT_INVALID',events[0]['reason'])

    def test_independent_finish_rejects_legacy_plan_even_with_current_run_version(self):
        self.allow_artifact.stop()
        current=self.reserve(); path=self.materialize(current)
        records=[json.loads(line) for line in path.read_text('utf-8').splitlines()]
        (path.parent/'old-plan.csv').write_bytes((path.parent/records[0]['schedule_csv']).read_bytes())
        records[0]['schedule_csv']='old-plan.csv'
        path.write_text('\n'.join(map(json.dumps,records))+'\n',encoding='utf-8')
        events=self.close(current)
        self.assertEqual(events[0]['event'],'INVALID')
        self.assertIn('exact versioned',events[0]['reason'])

    def test_independent_finish_rejects_changed_plan_generator_identity(self):
        self.allow_artifact.stop()
        current=self.reserve(); path=self.materialize(current)
        meta=json.loads(path.read_text('utf-8').splitlines()[0])
        plan_meta=(path.parent/meta['schedule_csv']).with_suffix('.meta.json')
        changed=json.loads(plan_meta.read_text('utf-8')); changed.pop('generator_version')
        plan_meta.write_text(json.dumps(changed),encoding='utf-8')
        self.assertEqual(self.close(current)[0]['event'],'INVALID')


class FreezeChecks(unittest.TestCase):
    def test_canonical_all_stops_after_one_deterministic_validation_failure(self):
        from sysbench import execute
        for reason in ('HOST_VALUEERROR','HOST_PRECLOCKSETUPERROR'):
            failure=dict(event='INVALID',reason=reason,attempt='0')
            with patch.object(execute,'maintenance'),patch.object(execute,'awake',return_value=nullcontext()),\
                    patch.object(execute,'next_slot',return_value=({'slot_id':'normal-001'},{})),\
                    patch.object(execute,'execute_slot',return_value=[failure]) as launch,redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'before spending'): execute.main(['--all','--maintenance-ready'])
                self.assertEqual(launch.call_count,1)

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
