"""Synthetic arrays/temporary artifacts only: no real IDs, traces or runs DB."""

import contextlib
import copy
import csv
import hashlib
import io
import multiprocessing
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

import v2_lite as runner


class LiteProperties(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = runner.load_config()
        cls.config["design"]["workers"] = 1  # Mocked IO never delegates to child processes.
        for field in ("confirm_eta_sensitivity", "known_delay_hpa"):
            cls.config["execution"].pop(field, None)  # Historical fixtures exercise default False.
        for field in ("design_human_review", "confirm_human_review", "confirm_numeric_review"):
            cls.config["gates"][field] = "PENDING"  # Gate properties remain synthetic after design approval.

    def toy(self, case="delay600"):
        y = np.full(3035, 8., dtype=float)
        y[3000:] = np.resize([1., 16., 8., 3.], 35)
        scenario = next(c for c in runner.scenarios(self.config) if c["name"] == case)
        return runner.prepare_episode(y, 100, 3010, len(y), scenario, self.config)

    def validation_rows(self, base="HPA"):
        return [dict(source="synthetic", app="toy", predictor="persistence", seed=0,
                     scenario="nominal", phase="validation", budget_mode="causal", base=base,
                     candidate=i, status="OK", risk=.1, cost=10. + i)
                for i in range(30)]

    def test_grids_and_default_plan_never_read_arrays(self):
        with mock.patch.object(runner.np, "load", side_effect=AssertionError("real array IO")):
            grid = runner.candidate_grid(self.config)
            old = runner.configurations(self.config)
            for base in runner.BASES[:3]:
                self.assertEqual(grid[base], old[base])
            self.assertEqual(len(grid["BACC-cal"]), 30)
            self.assertEqual({c["parameter"] for c in grid["BACC-cal"]}, {0.})
            counts = runner.plan(self.config)
            self.assertEqual((len(counts["primary"]), len(counts["descriptive_cases"]), counts["case_repetitions"]), (4, 13, 21))
            self.assertEqual((counts["explore_locked_slots_per_app"], counts["confirm_locked_slots_per_app"]), (100, 189))
            self.assertEqual(counts["exploration_cases"], list(runner.PRIMARY))
            self.assertEqual(len(runner.execution_cases(self.config, "confirm")), 13)
            with mock.patch.object(runner, "load_config", return_value=self.config), contextlib.redirect_stdout(io.StringIO()):
                runner.main([])

    def test_nominal_v1_function_and_invalid_selection_inputs(self):
        rows = self.validation_rows()
        rows[1].update(risk=.051, cost=3.)
        rows[2].update(risk=.04, cost=3.)
        selected = runner.select_nominal(rows, self.config, methods=("HPA",))["HPA"]
        self.assertEqual(selected["candidate"], 1)  # Equal feasible cost: index, not smaller risk.
        self.assertEqual(selected["config"], runner.candidate_grid(self.config)["HPA"][1])
        for row in rows:
            row.update(risk=.1, cost=10.)
        rows[5].update(risk=.06, cost=9.)
        rows[6].update(risk=.06, cost=8.)
        selected = runner.select_nominal(rows, self.config, methods=("HPA",))["HPA"]
        self.assertEqual((selected["candidate"], selected["validation_status"]), (6, "VALIDATION_BUDGET_NOT_MET"))
        for key, value in (("phase", "test"), ("wrapped", 1), ("predictor", "lightgbm")):
            bad = copy.deepcopy(rows)
            bad[0][key] = value
            with self.assertRaises(ValueError):
                runner.select_nominal(bad, self.config, methods=("HPA",))
        rows[0]["config"] = dict(risk=.05, parameter=999., window=0, stable=300)
        with self.assertRaises(ValueError):
            runner.select_nominal(rows, self.config, methods=("HPA",))
        unavailable = self.validation_rows("PAC-h")
        for row in unavailable:
            row.update(status="NO_RANK", risk=None, cost=None)
        self.assertEqual(runner.select_nominal(unavailable, self.config, methods=("PAC-h",))["PAC-h"]["validation_status"], "NO_VALID_CONFIG")

    def test_shared_configuration_proposals_lambda_and_conditional_bound(self):
        episode = self.toy()
        grid = runner.candidate_grid(self.config)
        for base in runner.BASES:
            cfg = grid[base][-1]
            saved = copy.deepcopy(cfg)
            proposal = runner.proposal_path(episode, base, cfg, self.config)
            original = proposal.copy()
            raw = runner.evaluate_episode(episode, base, cfg, self.config, proposals=proposal)
            self.assertIsNone(raw["lambda_final"])
            self.assertIsNone(raw["lambda_min"])
            for eta in self.config["search"]["wrapper_eta"]:
                wrapped = runner.evaluate_episode(episode, base, cfg, self.config, eta, proposal)
                self.assertEqual(wrapped["config"], raw["config"])
                self.assertAlmostEqual(wrapped["risk"], .05 - wrapped["lambda_final"] / (eta * wrapped["minutes"]))
                self.assertLessEqual(wrapped["lambda_min"], min(0., wrapped["lambda_final"]))
                self.assertEqual((wrapped["h_actual"], wrapped["dmax"], wrapped["L"], wrapped["K"]), (11, 11, 14, 280))
                self.assertLessEqual(wrapped["risk"], wrapped["cdt_bound"] + 1e-12)
            np.testing.assert_array_equal(proposal, original)
            self.assertEqual(cfg, saved)
        quiet = copy.deepcopy(episode)
        quiet["demand"][quiet["start"]:] = 0.
        quiet["physical_floor"], quiet["contract"] = 0., 1
        row = runner.evaluate_episode(quiet, "HPA", grid["HPA"][0], self.config, .1)
        self.assertGreater(row["lambda_final"], 0.)
        self.assertEqual(row["lambda_min"], 0.)

    def test_mature_prefix_and_future_noninterference(self):
        episode = self.toy("nominal")
        h, start, y = episode["h_assumed"], episode["start"], episode["demand"]
        scores = runner.calibration_scores(episode, 240)
        np.testing.assert_array_equal(scores, np.clip((y[start-240:start] + 1) / (y[start-240-h:start-h] + 1), .05, 20.))
        changed = copy.deepcopy(episode)
        changed["demand"][start+1:] = 999.
        for base in ("PAC-h", "BACC-cal"):
            cfg = runner.candidate_grid(self.config)[base][-1]
            original = runner.proposal_path(episode, base, cfg, self.config)
            future = runner.proposal_path(changed, base, cfg, self.config)
            self.assertEqual(original[0], future[0])

    def test_execution_contract_and_pac_unavailable(self):
        episode = self.toy()
        cfg = runner.candidate_grid(self.config)["Point+margin"][0]
        slower = copy.deepcopy(episode["case"])
        slower["delay"] = 1200
        outside = runner.prepare_episode(episode["demand"], 100, 3010, 3035, slower, self.config)
        row = runner.evaluate_episode(outside, "Point+margin", cfg, self.config, .1)
        self.assertEqual(row["contract"], 0)
        self.assertIsNone(row["cdt_bound"])
        over = episode["demand"].copy()
        over[3010:] = 33.
        outside = runner.prepare_episode(over, 100, 3010, 3035, episode["case"], self.config)
        self.assertGreater(outside["physical_floor"], 0)
        self.assertEqual(outside["contract"], 0)
        bad_pac = dict(risk=.01, parameter=0., window=2, stable=300)
        row = runner.evaluate_episode(episode, "PAC-h", bad_pac, self.config)
        self.assertEqual(row["status"], "NO_RANK")
        self.assertIsNone(row["paid"])
        with self.assertRaises(ValueError):
            runner.evaluate_episode(episode, "BACC-lite", runner.candidate_grid(self.config)["BACC-lite"][0], self.config, .1)

    def test_complete_synthetic_pipeline_and_primary_only_view(self):
        episode = self.toy()
        rows = list(runner.application_pipeline(episode["demand"], 100, 3000, 3010, self.config))
        candidates = [r for r in rows if r["role"] == "candidate"]
        tests = [r for r in rows if r["phase"] == "test"]
        self.assertEqual((len(candidates), len(tests), len(rows)), (150, 100, 255))
        self.assertEqual({r["scenario"] for r in tests}, set(runner.PRIMARY))
        locked = {r["base"]: r for r in rows if r["phase"] == "validation" and r["role"] == "locked"}
        self.assertEqual(set(locked), set((*runner.BASES, runner.COMPETITOR)))
        for row in tests:
            self.assertEqual((row["candidate"], row["config"]), (locked[row["base"]]["candidate"], locked[row["base"]]["config"]))
        primary = runner.primary_rows(rows, self.config)
        self.assertEqual(len(primary), 100)
        poison = [dict(r, paid=1e99) if r["scenario"] not in runner.PRIMARY else dict(r) for r in rows]
        self.assertEqual(primary, runner.primary_rows(poison, self.config))
        confirmation = copy.deepcopy(self.config)
        confirmation["frozen_numeric"]["selected_eta"] = .1
        confirmed = list(runner.application_pipeline(episode["demand"], 100, 3000, 3010, confirmation, "confirm"))
        confirmed_tests = [r for r in confirmed if r["phase"] == "test"]
        self.assertEqual(len(confirmed_tests), 189)
        self.assertEqual(len({r["scenario"] for r in confirmed_tests}), 13)
        self.assertEqual({r["replicate"] for r in confirmed_tests if r["scenario"] == "flash3x15"},
                         set(self.config["design"]["seeds"]))

    def test_unavailable_base_emits_both_versions_without_substitution(self):
        episode = self.toy()
        select = runner.select_nominal
        def unavailable(rows, config, methods):
            if methods == ("PAC-h",):
                return {"PAC-h": dict(candidate=None, config=None, validation_status="NO_VALID_CONFIG")}
            return select(rows, config, methods)
        with mock.patch.object(runner, "select_nominal", side_effect=unavailable):
            rows = list(runner.application_pipeline(episode["demand"], 100, 3000, 3010, self.config))
        pac = [r for r in rows if r["phase"] == "test" and r["base"] == "PAC-h"]
        self.assertEqual(len(pac), 24)
        self.assertEqual({r["status"] for r in pac}, {"NO_VALID_CONFIG"})
        self.assertEqual({r["wrapped"] for r in pac}, {0, 1})
        self.assertTrue(all(r["candidate"] is None and "paid" not in r for r in pac))

    def test_known_delay_hpa_uses_validation_only_and_never_changes_nominal_lock(self):
        episode = self.toy()
        with mock.patch.object(runner, "select_nominal", wraps=runner.select_nominal) as select:
            rows = list(runner.known_delay_hpa_pipeline(episode["demand"], 100, 3000, 3010, self.config))
        self.assertEqual(select.call_count, 4)
        self.assertEqual(len(rows), 128)
        for call in select.call_args_list:
            inputs = call.args[0]
            self.assertEqual(len(inputs), 30)
            self.assertEqual({r["phase"] for r in inputs}, {"validation"})
            self.assertEqual({r["scenario"] for r in inputs}, {"nominal"})
            self.assertEqual(len({r["h_actual"] for r in inputs}), 1)
        self.assertEqual({r["h_actual"] for r in rows}, {3, 4, 6, 11})
        changed = episode["demand"].copy()
        changed[3010:] = 999.  # Test truths cannot affect oracle validation selection.
        future = list(runner.known_delay_hpa_pipeline(changed, 100, 3000, 3010, self.config))
        validation = [r for r in rows if r["phase"] == "validation"]
        self.assertEqual(validation, [r for r in future if r["phase"] == "validation"])
        chosen = {r["scenario"]: r for r in validation if r["role"] == "oracle_locked"}
        for point in [r for r in future if r["phase"] == "test"]:
            self.assertEqual(point["role"], "oracle_locked")
            self.assertEqual((point["candidate"], point["config"]),
                             (chosen[point["scenario"]]["candidate"], chosen[point["scenario"]]["config"]))
            self.assertEqual((point["base"], point["wrapped"]), ("HPA", 0))
        self.assertEqual(runner.primary_rows(rows, self.config), [])

    def test_known_delay_unavailable_has_explicit_validation_and_test_placeholders(self):
        episode = self.toy()
        unavailable = {"HPA": dict(candidate=None, config=None, validation_status="NO_VALID_CONFIG")}
        with mock.patch.object(runner, "select_nominal", return_value=unavailable), \
             mock.patch.object(runner, "evaluate_episode", wraps=runner.evaluate_episode) as evaluate:
            rows = list(runner.known_delay_hpa_pipeline(episode["demand"], 100, 3000, 3010, self.config))
        locked = [r for r in rows if r["role"] == "oracle_locked"]
        self.assertEqual((len(rows), len(locked), evaluate.call_count), (128, 8, 120))
        self.assertEqual({r["phase"] for r in locked}, {"validation", "test"})
        self.assertTrue(all(r["status"] == "NO_VALID_CONFIG" and r["candidate"] is None and "paid" not in r
                            for r in locked))

    def test_confirm_descriptions_preserve_main_eta_and_historical_exploration(self):
        episode = self.toy()
        historical = copy.deepcopy(self.config)
        historical["frozen_numeric"]["selected_eta"] = .1
        reference = list(runner.application_pipeline(episode["demand"], 100, 3000, 3010, historical, "confirm"))
        cfg = copy.deepcopy(historical)
        cfg["execution"].update(confirm_eta_sensitivity=True, known_delay_hpa=True)
        saved = copy.deepcopy(cfg)
        with mock.patch.object(runner, "select_nominal", wraps=runner.select_nominal) as select:
            rows = list(runner.application_pipeline(episode["demand"], 100, 3000, 3010, cfg, "confirm"))
        self.assertEqual(select.call_count, 9)  # Five nominal bases plus four validation-only HPA oracles.
        self.assertEqual(cfg, saved)
        main = [r for r in rows if r["role"] in ("candidate", "locked")
                and (not r["wrapped"] or r["eta"] == .1)]
        self.assertEqual(main, reference)
        self.assertEqual((len(rows), len([r for r in rows if r["phase"] == "test" and r["role"] == "locked"])),
                         (808, 525))
        self.assertEqual({r["eta"] for r in rows if r["wrapped"]}, set(cfg["search"]["wrapper_eta"]))
        counts = runner.plan(cfg)
        self.assertEqual((counts["confirm_total_slots_per_app"], counts["confirm_total_slots_expected"],
                          counts["confirm_oracle_slots_per_app"]), (808, 175336, 128))
        before = list(runner.application_pipeline(episode["demand"], 100, 3000, 3010, historical, "explore"))
        after = list(runner.application_pipeline(episode["demand"], 100, 3000, 3010, cfg, "explore"))
        self.assertEqual(after, before)
        self.assertTrue(all(r["role"] in ("candidate", "locked") for r in after))
        cfg["frozen_numeric"]["selected_eta"] = -1
        with mock.patch.object(runner, "evaluate_episode", side_effect=AssertionError("Unselected confirm started")):
            with self.assertRaisesRegex(ValueError, "not selected"):
                list(runner.application_pipeline(episode["demand"], 100, 3000, 3010, cfg, "confirm"))

    def test_complete_confirm_task_with_both_descriptions_uses_parent_database_writer(self):
        episode = self.toy()
        cfg = copy.deepcopy(self.config)
        cfg["frozen_numeric"]["selected_eta"] = .1
        cfg["execution"].update(confirm_eta_sensitivity=True, known_delay_hpa=True)
        task = ("synthetic", "toy", episode["demand"], 100, 3000, 3010, cfg, "confirm", None)
        expected = runner.application_task(task)
        with runner.ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
            actual = pool.submit(runner.application_task, task).result()
        self.assertEqual(actual, expected)
        source, app, rows = actual
        with tempfile.TemporaryDirectory(dir=runner.ROOT / "work", prefix="v2-lite-property-") as folder:
            path = Path(folder)
            db = runner.output_database(path / "synthetic.sqlite", "confirm", "synthetic-ref")
            try:
                with db:
                    for point in rows:
                        row = dict(point, stage="confirm", config_ref="synthetic-ref", source=source, app=app)
                        db.execute("INSERT INTO runs VALUES (" + ",".join("?" for _ in runner.FIELDS) + ")",
                                   [row.get(field) for field in runner.FIELDS])
                    db.execute("INSERT INTO jobs VALUES (?,?,?,?)", ("confirm", "synthetic-ref", source, app))
                self.assertEqual(db.execute("SELECT COUNT(*) FROM runs").fetchone()[0], 808)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM runs WHERE role='oracle_candidate'").fetchone()[0], 120)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM runs WHERE phase='test' AND role='oracle_locked'").fetchone()[0], 4)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 1)
                runner.export_summary(db, path / "synthetic.csv")
            finally:
                db.close()
            with (path / "synthetic.csv").open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 808)

    def test_workload_arithmetic_counts_shared_preparation_and_all_confirmation_cases(self):
        from scripts.benchmark_v2_lite import workload
        meta = {source: dict(applications=50, validation=4320, test=minutes)
                for source, minutes in zip(runner.SOURCES, (36000, 25920, 8640))}
        timing = {(base, operation): dict(median_seconds=1., max_seconds=2., measured_ticks=1000)
                  for base in (*runner.BASES, runner.COMPETITOR)
                  for operation in (("prepare", "raw", "wrapped") if base in runner.BASES else ("raw",))}
        rows = workload(self.config, meta, timing, 3.)
        explore = [r for r in rows if r["stage"] == "explore"]
        test = [r for r in explore if r["notes"].startswith("test") and r["operation"] != "prepare"]
        self.assertEqual(sum(r["total_ticks"] for r in test), 352800000)
        self.assertEqual(sum(r["total_ticks"] for r in explore), 467568000)
        self.assertTrue(all(r["calls"] == 0 for r in explore if r["base"] in runner.BASES[:3]
                            and r["notes"].startswith("validation")))
        confirm = [r for r in rows if r["stage"] == "confirm"]
        self.assertEqual(sum(r["total_ticks"] for r in confirm), 1568095200)
        self.assertEqual({r["case_repetitions"] for r in confirm}, {21})
        for row in rows:
            self.assertAlmostEqual(row["conservative_seconds"], row["estimated_seconds"] * 6)

    def test_two_synthetic_spawn_tasks_match_sequential_application_outputs(self):
        episode = self.toy()
        task = ("synthetic", "toy", episode["demand"], 100, 3000, 3010, self.config, "explore", None)
        other = ("synthetic", "toy-second", episode["demand"] * .5, 100, 3000, 3010, self.config, "explore", None)
        expected = list(map(runner.application_task, (task, other)))
        with runner.ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
            actual = list(pool.map(runner.application_task, (task, other), chunksize=1))
        self.assertEqual(actual, expected)
        changed = copy.deepcopy(self.config)
        changed["design"]["workers"] = 8
        self.assertEqual(runner.plan(changed)["workers"], 8)
        changed["design"]["workers"] = 9
        with self.assertRaises(ValueError):
            runner.plan(changed)

    def test_stage_gates_precede_every_real_data_or_database_call(self):
        for stage in ("explore", "confirm"):
            with mock.patch.object(runner, "load_source", side_effect=AssertionError("NPZ IO")), \
                 mock.patch.object(runner, "load_reuse", side_effect=AssertionError("reuse IO")), \
                 mock.patch.object(runner, "output_database", side_effect=AssertionError("SQLite IO")):
                with self.assertRaises(RuntimeError):
                    runner.run_stage(self.config, stage)
        approved = copy.deepcopy(self.config)
        approved["gates"].update(design_human_review="APPROVED", confirm_human_review="APPROVED", confirm_numeric_review="APPROVED")
        approved["frozen_numeric"]["selected_eta"] = .1
        approved["design"]["deadline"] = "2099-01-01T00:00:00+00:00"
        def fake_git(args, **kwargs):
            if args[1] == "cat-file":
                return ""
            return "commit\n" if args[-1].endswith("^{commit}") else "blob\n"
        with mock.patch.object(runner.subprocess, "check_output", side_effect=fake_git) as calls:
            self.assertEqual(runner.require_stage_gate(approved, "confirm"), "commit")
            self.assertEqual(sum(c.args[0][1] == "cat-file" for c in calls.call_args_list), len(runner.CRITICAL))
        def changed_git(args, **kwargs):
            return "changed\n" if args[1] == "hash-object" else fake_git(args, **kwargs)
        with mock.patch.object(runner.subprocess, "check_output", side_effect=changed_git):
            with self.assertRaisesRegex(RuntimeError, "Frozen source differs"):
                runner.require_stage_gate(approved, "explore")

    def test_prepared_npz_loader_and_confirmation_identity_guard(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT / "work", prefix="v2-lite-property-") as folder:
            root = Path(folder)
            old, confirm = root / "old", root / "confirm"
            old.mkdir(); confirm.mkdir()
            source = runner.SOURCES[0]
            cfg = copy.deepcopy(self.config)
            cfg["io"]["exploration_data"] = "old"
            cfg["io"]["confirmation_data"] = "confirm"
            cfg["design"]["exploration_apps_per_trace"] = 1
            cfg["design"]["confirmation_cap_per_trace"] = 1
            prefix = cfg["selection"]["hash_prefix"]
            expected = min(("a", "b"), key=lambda app: hashlib.sha256((prefix + app).encode()).digest())
            other, third = runner.SOURCES[1:]
            # A bare ID in another source is not an exclusion for this source.
            (old / "selected-unit-ids.csv").write_text(
                f"trace,unit_id\n{source},old\n{other},{expected}\n{third},old\n", encoding="utf-8")
            (old / "eligible-unit-pool.csv").write_text(
                f"source,unit_id\n{source},old\n{source},a\n{source},b\n{other},{expected}\n{other},new1\n{third},old\n{third},new2\n", encoding="utf-8")
            def save(directory, app):
                np.savez(directory / f"{source}.npz", demand=np.ones((1, 11525)), unit_id=np.array([app]),
                         fit_end=4320, cal_end=7200, val_end=11520)
            save(old, "old")
            with mock.patch.object(runner, "ROOT", root), mock.patch.dict(runner.CONFIRM_COUNTS, dict.fromkeys(runner.SOURCES, 1)):
                self.assertEqual(runner.load_source(old, source, cfg, "explore")["unit_id"], ["old"])
                save(confirm, "old")
                tail = f"{other},new1\n{third},new2\n"
                (confirm / "selected-unit-ids.csv").write_text(f"trace,unit_id\n{source},old\n" + tail, encoding="utf-8")
                with self.assertRaises(ValueError):
                    runner.load_source(confirm, source, cfg, "confirm")
                save(confirm, expected)
                (confirm / "selected-unit-ids.csv").write_text(f"trace,unit_id\n{source},{expected}\n" + tail, encoding="utf-8")
                self.assertEqual(runner.load_source(confirm, source, cfg, "confirm")["unit_id"], [expected])
                with mock.patch.object(runner.np, "load", side_effect=AssertionError("identity check opened NPZ")):
                    self.assertEqual(runner.validate_manifest(cfg, "confirm")[source], (expected,))
                (confirm / "selected-unit-ids.csv").write_text(f"trace,unit_id\n{source},{expected}\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    runner.validate_manifest(cfg, "confirm")
            with self.assertRaisesRegex(RuntimeError, "NOT_PREPARED"):
                runner.load_source(confirm, "missing", cfg, "confirm")

    def test_sqlite_checkpoint_and_export_are_small_synthetic_only(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT / "work", prefix="v2-lite-property-") as folder:
            path = Path(folder) / "synthetic.sqlite"
            db = runner.output_database(path, "explore", "synthetic-ref")
            row = {k: None for k in runner.FIELDS}
            row.update(stage="explore", config_ref="synthetic-ref", source="synthetic", app="toy",
                       phase="test", role="locked", base="HPA", wrapped=0, candidate=0, status="OK",
                       violations=1, minutes=20, paid=40, ready=30, actions=2, contract=1)
            with db:
                db.execute("INSERT INTO runs VALUES (" + ",".join("?" for _ in runner.FIELDS) + ")", list(row.values()))
                db.execute("INSERT INTO jobs VALUES (?,?,?,?)", ("explore", "synthetic-ref", "synthetic", "toy"))
            output = Path(folder) / "synthetic.csv"
            runner.export_summary(db, output)
            with output.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual((rows[0]["wrapped"], rows[0]["lambda_final"]), ("0", ""))
            db.close()
            with self.assertRaisesRegex(RuntimeError, "different frozen reference"):
                runner.output_database(path, "explore", "different-ref")

    def test_cache_requires_audit_values_and_the_entire_matching_grid(self):
        legacy_text = (runner.ROOT / "configs/gate2.toml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory(dir=runner.ROOT / "work", prefix="v2-lite-property-") as folder:
            root = Path(folder)
            (root / "configs").mkdir()
            (root / "configs/gate2.toml").write_text(legacy_text, encoding="utf-8")
            cfg = copy.deepcopy(self.config)
            cfg["io"]["reuse_csv"] = "reuse.csv"
            params = runner.candidate_grid(cfg)["HPA"][0]
            row = dict(trace=runner.SOURCES[0], app="toy", method="HPA", candidate=0,
                       audit_status="PASS", reuse_eligible=1, predictor="persistence", seed=0,
                       phase="validation", scenario="nominal", budget_mode="causal", status="OK",
                       candidate_slots=30, candidate_match=1, validation_match=1, config_match=1,
                       validation_cutoff=.051, reference_candidate=0, v1_candidate=0, v2_candidate=0,
                       validation_risk=.04, validation_cost=3., locked_validation_risk=.04,
                       locked_validation_cost=3., **params)
            def write(point):
                with (root / "reuse.csv").open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(point))
                    writer.writeheader(); writer.writerow(point)
            write(row)
            with mock.patch.object(runner, "ROOT", root):
                self.assertEqual(len(runner.load_reuse(cfg)), 1)
                changed = copy.deepcopy(cfg)
                changed["search"]["hpa_utilization"][-1] = 1.3  # Selected index 0 itself still matches.
                self.assertEqual(runner.load_reuse(changed), {})
                write(dict(row, audit_status="PENDING"))
                self.assertEqual(runner.load_reuse(cfg), {})

    def test_real_io_loop_checkpoint_with_only_temporary_synthetic_inputs(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT / "work", prefix="v2-lite-property-") as folder:
            root = Path(folder)
            prepared = root / "prepared"
            prepared.mkdir()
            for source in runner.SOURCES:
                (prepared / f"{source}.npz").touch()  # Loader is mocked: these are never opened as NPZ.
            cfg = copy.deepcopy(self.config)
            cfg["io"].update(exploration_data="prepared", output_db="synthetic.sqlite", summary_csv="synthetic.csv")
            fake_data = dict(unit_id=["toy"], demand=np.ones((1, 3035)), fit_end=100, cal_end=3000, val_end=3010)
            point = dict(scenario="delay120", replicate=0, phase="test", role="locked", base="HPA",
                         wrapped=0, candidate=0, status="OK", violations=1, minutes=25, paid=50,
                         ready=40, actions=2, contract=1)
            with mock.patch.object(runner, "ROOT", root), \
                 mock.patch.object(runner, "require_stage_gate", return_value="synthetic-ref"), \
                 mock.patch.object(runner, "load_source", return_value=fake_data), \
                 mock.patch.object(runner, "load_reuse", return_value={}), \
                 mock.patch.object(runner, "application_pipeline", side_effect=lambda *a: iter([dict(point)])) as pipeline, \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.run_stage(cfg, "explore")
                runner.run_stage(cfg, "explore")
                self.assertEqual(pipeline.call_count, 3)  # The second stage call resumes without duplicate rows.
            with runner.sqlite3.connect(root / "synthetic.sqlite") as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM runs").fetchone()[0], 3)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 3)
            db.close()
            with (root / "synthetic.csv").open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 3)


if __name__ == "__main__":
    unittest.main()
