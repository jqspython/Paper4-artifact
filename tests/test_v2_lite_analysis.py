"""Counterexamples and boundary checks on synthetic aggregate episodes only."""

import copy
import unittest
from unittest import mock

import numpy as np

from scripts import analyze_v2_lite as analysis
from scripts import report_v2_lite_confirmation as confirmation
from v2_lite import BASES, PRIMARY, SOURCES, candidate_grid, load_config


class EstimandProperties(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.config["gate"]["minimum_contract_apps_per_trace"] = 1
        self.config["statistics"].update(bootstrap_repetitions=30, maximum_bootstrap_attempts=150)

    def toy(self, counts=(5, 7, 9)):
        data = {}
        for s, n in zip(SOURCES, counts):
            values = np.empty((n, 4, 4, 2, 2))
            values[..., 0] = .05
            values[:, :, :2, 0, 0] = .08
            values[:, :, 2:, 0, 0] = .03
            values[..., 1] = 10.
            values[:, :, :2, 1, 1] = 12.
            values[:, :, 2:, 1, 1] = 9.
            data[s] = dict(ids=tuple(f"toy-{i}" for i in range(n)), values=values,
                           available=np.ones((n, 4), dtype=bool), validation_ok=np.ones((n, 4), dtype=bool),
                           contract=np.ones((n, 4), dtype=int))
        return data

    def result_rows(self, data):
        result = []
        for source, panel in data.items():
            for i, app in enumerate(panel["ids"]):
                for c, case in enumerate(PRIMARY):
                    for b, base in enumerate(BASES):
                        for w in (0, 1):
                            risk, cost = panel["values"][i, c, b, w]
                            result.append(dict(stage="explore", phase="test", role="locked", source=source,
                                               app=app, scenario=case, replicate=0, base=base, wrapped=w,
                                               eta=.05 if w else None, status="OK", minutes=1000,
                                               validation_status="OK",
                                               violations=round(risk * 1000), paid=cost * 1000,
                                               contract=int(panel["contract"][i, c]), candidate=3,
                                               config_risk=.05, parameter=1., window=240, stable=300))
        return result

    def test_vendor_weights_and_no_unauthorized_vendor_cost_guard(self):
        data = self.toy((2, 6, 13))
        for source, ratio in zip(SOURCES, (1.6, 1.2, 1.1)):
            data[source]["values"][:, :, :2, 1, 1] = 10 * ratio
        point = analysis.estimate(data, self.config)
        self.assertAlmostEqual(point["vendors"]["Huawei"]["p1_cost_ratio"], 1.4)
        self.assertAlmostEqual(point["point"]["p1_cost_ratio"], 1.25)
        interval = analysis.bootstrap(data, self.config)
        self.assertEqual(analysis.verdict(point, interval, self.config)["P1"], "PASS")
        for panel in data.values():
            panel["values"][:, :, :2, 1, 1] = 12.6
        point = analysis.estimate(data, self.config)
        self.assertEqual(analysis.verdict(point, interval, self.config)["P1"], "ONLY_BY_OVERPROVISIONING")

    def test_source_medians_and_mixture_can_reverse(self):
        data = self.toy((5, 5, 5))
        data[SOURCES[0]]["values"][:, :, :, 1, 1] = 14.
        data[SOURCES[1]]["values"][:, :, :, 1, 1] = np.array([1., 1., 1., 10., 10.])[:, None, None]
        data[SOURCES[2]]["values"][:, :, :, 1, 1] = 10.
        primary = analysis.estimate(data, self.config)["point"]["p2_cost_ratio"]
        mixed = analysis.estimate(data, self.config, "mixed_median")["point"]["p2_cost_ratio"]
        self.assertAlmostEqual(primary, .875)
        self.assertAlmostEqual(mixed, 1.1)

    def test_pairing_missing_contract_and_configuration_guards(self):
        data = self.toy()
        expected = {s: p["ids"] for s, p in data.items()}
        rows = self.result_rows(data)
        rebuilt = analysis.panels(rows, expected, .05, "explore")
        for source in SOURCES:
            np.testing.assert_allclose(rebuilt[source]["values"], data[source]["values"])
        for mutation in (lambda r: r.pop(), lambda r: r.append(dict(r[0])),
                         lambda r: r[1].update(contract=0), lambda r: r[1].update(candidate=4),
                         lambda r: r[1].update(status="FAILED_RUN"), lambda r: r[1].update(app="unexpected")):
            changed = copy.deepcopy(rows)
            mutation(changed)
            with self.assertRaises(analysis.Incomplete):
                analysis.panels(changed, expected, .05, "explore")

    def test_fixed_source_bootstrap_reproducible_and_preserves_pairs(self):
        data = self.toy((3, 11, 4))
        a = analysis.bootstrap(data, self.config)
        b = analysis.bootstrap(data, self.config)
        self.assertEqual(a, b)
        self.assertEqual(a["valid"], 30)
        self.assertEqual(a["intervals"]["p1_delta"], (1., 1.))
        self.assertEqual(analysis.bootstrap(data, self.config, "Huawei_source_slots")["status"], "OK")

    def test_individual_wrapper_floor_overrides_good_macro(self):
        data = self.toy((10, 10, 10))
        for panel in data.values():
            panel["values"][:2, :, 0, 1, 0] = .08
        result = analysis.estimate(data, self.config)
        self.assertGreater(result["point"]["p3_compliance"], .9)
        self.assertAlmostEqual(result["point"]["wrapper_HPA"], .8)
        self.assertEqual(analysis.verdict(result, analysis.bootstrap(data, self.config), self.config)["P3"], "FAIL")
        self.assertEqual(analysis.verdict(result, dict(status="CI_UNAVAILABLE"), self.config)["decision"], "STOP_WRAPPER_SAFETY")

    def test_all_nine_sensitivities_reported_even_when_not_assessable(self):
        data = self.toy()
        rows, decision = analysis.report(data, self.config)
        self.assertEqual({row["variant"] for row in rows if row["variant"] in analysis.VARIANTS}, set(analysis.VARIANTS))
        self.assertEqual(decision["decision"], "BOTH_CATEGORIES")
        self.config["gate"]["minimum_contract_apps_per_trace"] = 30
        rows, decision = analysis.report(data, self.config)
        self.assertIsNone(decision)
        self.assertEqual({row["variant"] for row in rows if row["metric"] == "status"}, set(analysis.VARIANTS))
        self.assertEqual(sum(row["metric"] == "contract_apps" for row in rows), 12)
        self.assertEqual(len(analysis.VARIANTS), 9)

    def unavailable_rows(self, n=20, missing=(0,), base="PAC-h"):
        data = self.toy((n, n, n))
        rows = self.result_rows(data)
        for row in rows:
            if row["source"] == SOURCES[0] and row["app"] in {f"toy-{i}" for i in missing} and row["base"] == base:
                row.update(status="NO_VALID_CONFIG", validation_status="NO_VALID_CONFIG", candidate=None,
                           violations=None, paid=None)
        expected = {s: p["ids"] for s, p in data.items()}
        return rows, expected

    def test_no_valid_config_is_an_entire_pair_and_exact_five_percent_is_allowed(self):
        rows, expected = self.unavailable_rows()
        rebuilt = analysis.panels(rows, expected, .05, "explore")
        self.assertFalse(rebuilt[SOURCES[0]]["available"][0, 2])
        self.assertTrue(np.isnan(rebuilt[SOURCES[0]]["values"][0, :, 2]).all())
        result = analysis.estimate(rebuilt, self.config)
        metrics = result["sources"][SOURCES[0]]
        self.assertEqual(metrics["unavailable"][2], 1)
        self.assertEqual(metrics["unavailable_fraction"][2], .05)
        np.testing.assert_array_equal(metrics["effective_support"][:, 2], 19)
        self.assertTrue(all(result["assessable"].values()))
        bad = copy.deepcopy(rows)
        bad.pop(next(i for i, row in enumerate(bad) if row["status"] == "NO_VALID_CONFIG"))
        with self.assertRaises(analysis.Incomplete):
            analysis.panels(bad, expected, .05, "explore")
        bad = copy.deepcopy(rows)
        next(r for r in bad if r["status"] == "NO_VALID_CONFIG")["validation_status"] = "OK"
        with self.assertRaises(analysis.Incomplete):
            analysis.panels(bad, expected, .05, "explore")

    def test_unavailability_over_five_percent_blocks_only_its_category_and_p3(self):
        rows, expected = self.unavailable_rows(missing=(0, 1))
        data = analysis.panels(rows, expected, .05, "explore")
        result = analysis.estimate(data, self.config)
        interval = dict(status="OK", intervals={"p1_delta": (.5, 1.)})
        decision = analysis.verdict(result, interval, self.config)
        self.assertEqual((decision["P1"], decision["P2"], decision["P3"]), ("PASS", "INCOMPLETE", "INCOMPLETE"))
        self.assertEqual(decision["decision"], "INCOMPLETE")
        self.assertIsNone(analysis.select_eta({eta: data for eta in self.config["search"]["wrapper_eta"]}, self.config))
        all_rows, all_expected = self.unavailable_rows(missing=tuple(range(20)))
        all_unavailable = analysis.panels(all_rows, all_expected, .05, "explore")
        test = next(t for t in analysis.paired_tests(all_unavailable, self.config)
                    if t["source"] == SOURCES[0] and t["base"] == "PAC-h")
        self.assertEqual(test["n"], 0)
        self.assertTrue(np.isnan(test["p"]) and np.isnan(test["holm_p"]))
        bad = copy.deepcopy(rows)
        next(r for r in bad if r["status"] == "OK")["status"] = "FAILED_RUN"
        with self.assertRaises(analysis.Incomplete):
            analysis.panels(bad, expected, .05, "explore")

    def test_bootstrap_carries_original_availability_not_resample_gate(self):
        rows, expected = self.unavailable_rows()
        data = analysis.panels(rows, expected, .05, "explore")
        original, above = analysis.estimate, []
        def tracked(sample, config, variant="primary", source_slots=None):
            result = original(sample, config, variant, source_slots)
            above.append(result["sources"][SOURCES[0]]["unavailable_fraction"][2] > .05)
            return result
        with mock.patch.object(analysis, "estimate", side_effect=tracked):
            interval = analysis.bootstrap(data, self.config)
        self.assertEqual((interval["status"], interval["valid"]), ("OK", 30))
        self.assertTrue(any(above))

    def test_validation_budget_sensitivity_excludes_pairs_and_empty_is_na(self):
        data = self.toy((20, 20, 20))
        data[SOURCES[0]]["validation_ok"][:, 0] = False
        primary = analysis.estimate(data, self.config)
        sensitivity = analysis.estimate(data, self.config, "validation_budget_met")
        self.assertTrue(primary["assessable"]["P1"])
        self.assertFalse(sensitivity["assessable"]["P1"])
        self.assertTrue(sensitivity["assessable"]["P2"])
        self.assertTrue(np.isnan(sensitivity["point"]["p1_delta"]))
        self.assertEqual(sensitivity["sources"][SOURCES[0]]["unavailable"][0], 0)
        self.assertEqual(sensitivity["sources"][SOURCES[0]]["validation_excluded"][0], 20)
        interval = analysis.bootstrap(data, self.config, "validation_budget_met")
        self.assertEqual(interval["status"], "PARTIAL")
        self.assertEqual(interval["metric_status"]["p1_delta"], "CI_UNAVAILABLE")
        self.assertEqual(interval["metric_status"]["p2_cost_ratio"], "OK")

    def test_fully_unavailable_o_base_preserves_complete_u_confidence_interval(self):
        rows, expected = self.unavailable_rows(missing=tuple(range(20)))
        data = analysis.panels(rows, expected, .05, "explore")
        result = analysis.estimate(data, self.config)
        interval = analysis.bootstrap(data, self.config)
        self.assertEqual(interval["status"], "PARTIAL")
        self.assertEqual(interval["metric_status"]["p1_delta"], "OK")
        self.assertEqual(interval["metric_status"]["p2_cost_ratio"], "CI_UNAVAILABLE")
        self.assertEqual(interval["metric_valid"]["p1_delta"], 30)
        decision = analysis.verdict(result, interval, self.config)
        self.assertEqual((decision["P1"], decision["P2"], decision["P3"]), ("PASS", "INCOMPLETE", "INCOMPLETE"))
        output, _ = analysis.report(data, self.config)
        def metric(name):
            return next(r for r in output if r["variant"] == "primary" and r["scope"] == "overall" and r["metric"] == name)
        self.assertEqual(metric("p1_delta")["ci_status"], "OK")
        self.assertEqual(metric("p2_cost_ratio")["ci_status"], "CI_UNAVAILABLE")
        self.assertEqual(metric("p2_cost_ratio")["bootstrap_valid"], 0)

    def test_total_spend_and_tail_can_reverse_good_median(self):
        data = self.toy((5, 5, 5))
        for panel in data.values():
            panel["values"][..., 1] = 2.
            panel["values"][:, :, :, 1, 1] = 1.
            panel["values"][0, :, :, 0, 1] = 1.
            panel["values"][0, :, :, 1, 1] = 16.
            panel["paid"] = panel["values"][..., 1] * 1000
        primary = analysis.estimate(data, self.config)
        total = analysis.estimate(data, self.config, "total_cost")
        self.assertAlmostEqual(primary["point"]["p2_cost_ratio"], .5)
        self.assertAlmostEqual(total["point"]["p2_cost_ratio"], 20 / 9)
        self.assertEqual(total["vendors"]["Huawei"]["p2_app_total_ratio_p90"], 16.)
        source = analysis.source_rows(total, "total_cost", self.config)
        row = next(r for r in source if r["scope"] == f"{SOURCES[0]}/PAC-h" and r["metric"] == "app_total_cost_ratio_p90")
        self.assertAlmostEqual(row["value"], 9.8)

    def test_total_spend_has_no_hidden_source_or_cohort_weight(self):
        data = self.toy((2, 6, 13))
        for source, base, wrapped in zip(SOURCES, (1., 10., 10.), (2., 5., 5.)):
            data[source]["values"][..., 1] = base
            data[source]["values"][:, :, :, 1, 1] = wrapped
            data[source]["paid"] = data[source]["values"][..., 1] * 1000
        result = analysis.estimate(data, self.config, "total_cost")
        huawei = (2 * 2 + 6 * 5) / (2 * 1 + 6 * 10)
        self.assertAlmostEqual(result["vendors"]["Huawei"]["p2_cost_ratio"], huawei)
        self.assertAlmostEqual(result["point"]["p2_cost_ratio"], (huawei + .5) / 2)

    def test_exploration_reports_all_points_and_grid_boundary_without_expansion(self):
        datasets = {eta: self.toy() for eta in self.config["search"]["wrapper_eta"]}
        selected = analysis.select_eta(datasets, self.config)
        rows = analysis.exploration_rows(datasets, self.config, selected)
        self.assertEqual({r["variant"] for r in rows}, {f"eta_{eta}" for eta in datasets})
        self.assertEqual(sum(r["metric"] == "selection_eligible" for r in rows), 5)
        self.assertEqual(sum(r["value"] for r in rows if r["metric"] == "grid_boundary_limitation"), 1)
        self.assertTrue(any(r["scope"] == "Huawei/PAC-h" and r["metric"] == "base_compliance" for r in rows))

    def test_eta_tie_break_and_safety_failure(self):
        datasets = {eta: self.toy() for eta in self.config["search"]["wrapper_eta"]}
        self.assertEqual(analysis.select_eta(datasets, self.config), .02)
        for data in datasets.values():
            for panel in data.values():
                panel["values"][..., 0] = .2
        self.assertIsNone(analysis.select_eta(datasets, self.config))

    def test_descriptive_flash_contract_is_and_without_seed_inflation(self):
        data = self.toy((1, 1, 1))
        expected = {s: p["ids"] for s, p in data.items()}
        rows = self.result_rows(data)
        for s in SOURCES:
            template = next(r for r in rows if r["source"] == s)
            for rep in self.config["design"]["seeds"]:
                rows.append(dict(template, scenario="flash3x15", replicate=rep, contract=int(rep != 71)))
        support = analysis.all_case_support(rows, expected, "explore", self.config)
        flash = [r for r in support if r["scope"] == f"{SOURCES[0]}/flash3x15"]
        self.assertEqual(next(r["value"] for r in flash if r["metric"] == "contract_apps"), 0)
        self.assertEqual(next(r["value"] for r in flash if r["metric"] == "selected_apps"), 1)
        self.assertEqual(next(r["value"] for r in flash if r["metric"] == "excluded_fraction"), 1)

    def confirmation_fixture(self, counts=(1, 1, 1)):
        """All records are artificial aggregate episodes, never trace IO."""
        data = self.toy(counts)
        expected = {s: p["ids"] for s, p in data.items()}
        self.config["frozen_numeric"]["selected_eta"] = .1
        self.config["execution"].update(known_delay_hpa=True, confirm_eta_sensitivity=True)
        rows = [dict(r, stage="confirm", eta=.1 if r["wrapped"] else None) for r in self.result_rows(data)]
        for source, apps in expected.items():
            for app in apps:
                for case in confirmation.CASES:
                    reps = self.config["design"]["seeds"] if case.startswith("flash") else (0,)
                    if case not in PRIMARY:
                        for base in BASES:
                            for wrapped in (0, 1):
                                template = next(r for r in rows if r["source"] == source and r["app"] == app
                                                and r["base"] == base and r["wrapped"] == wrapped)
                                rows.extend(dict(template, scenario=case, replicate=rep) for rep in reps)
                    template = next(r for r in rows if r["source"] == source and r["app"] == app
                                    and r["base"] == "HPA" and not r["wrapped"])
                    rows.extend(dict(template, base="BACC-lite", scenario=case, replicate=rep) for rep in reps)
                for case in PRIMARY:
                    candidates = []
                    for index, cfg in enumerate(candidate_grid(self.config)["HPA"]):
                        candidates.append(dict(template, scenario=case, replicate=0, phase="validation",
                            role="oracle_candidate", candidate=index, config_risk=cfg["risk"],
                            parameter=cfg["parameter"], window=cfg["window"], stable=cfg["stable"],
                            violations=40, paid=2000 + index * 1000))
                    rows.extend(candidates)
                    lock = dict(candidates[0], role="oracle_locked")
                    rows.extend((lock, dict(lock, phase="test", violations=20)))
        primary_wrapped = [r for r in rows if r["role"] == "locked" and r["wrapped"]]
        for eta in self.config["search"]["wrapper_eta"]:
            if eta != .1:
                rows.extend(dict(r, eta=eta, violations=1000 if eta == .5 else r["violations"]) for r in primary_wrapped)
        return rows, expected

    def test_confirmation_actual_risk_quantiles_thresholds_and_huawei_pooling(self):
        rows, expected = self.confirmation_fixture((1, 3, 1))
        for row in rows:
            if row["phase"] == "test" and row["role"] == "locked" and row["base"] == "HPA" and row["scenario"] in PRIMARY:
                if not row["wrapped"]:
                    row["violations"] = (50, 60, 61, 40)[PRIMARY.index(row["scenario"])] if row["source"] == SOURCES[0] else 40
        data = analysis.panels(rows, expected, .1, "confirm")
        described = confirmation.risk_distribution(data, self.config)
        raw = next(r for r in described if r["scope"] == "Huawei" and r["controller"] == "HPA_raw")
        wrapped = next(r for r in described if r["scope"] == "Huawei" and r["controller"] == "HPA_wrapped")
        self.assertEqual((raw["N_app_case_cells"], raw["distinct_apps"]), (16, 4))
        self.assertAlmostEqual(raw["q50"], .04)
        self.assertAlmostEqual(raw["fraction_risk_gt_005"], 2 / 16)
        self.assertAlmostEqual(raw["fraction_risk_gt_006"], 1 / 16)
        self.assertEqual((wrapped["fraction_risk_gt_005"], wrapped["fraction_risk_gt_006"]), (0., 0.))
        self.assertEqual(len(described), 40)

    def test_confirmation_risk_uses_common_available_pairs_and_contract(self):
        rows, expected = self.confirmation_fixture((1, 3, 1))
        for row in rows:
            if row["phase"] != "test" or row["role"] != "locked":
                continue
            if row["source"] == SOURCES[1] and row["app"] == "toy-0" and row["base"] == "PAC-h":
                row.update(status="NO_VALID_CONFIG", validation_status="NO_VALID_CONFIG", candidate=None,
                           violations=None, paid=None)
            if row["source"] == SOURCES[0] and row["scenario"] == PRIMARY[0]:
                row["contract"] = 0
        data = analysis.panels(rows, expected, .1, "confirm")
        described = confirmation.risk_distribution(data, self.config)
        pac = [r for r in described if r["scope"] == "Huawei" and r["base"] == "PAC-h"]
        hpa = [r for r in described if r["scope"] == "Huawei" and r["base"] == "HPA"]
        self.assertEqual([r["N_app_case_cells"] for r in pac], [11, 11])
        self.assertEqual([r["N_app_case_cells"] for r in hpa], [15, 15])
        self.assertEqual([r["unavailable_apps"] for r in pac], [1, 1])

    def test_known_delay_reference_is_validation_locked_and_never_primary(self):
        rows, expected = self.confirmation_fixture()
        original = analysis.panels(rows, expected, .1, "confirm")
        main_only = [r for r in rows if r["role"] == "locked"]
        filtered = analysis.panels(main_only, expected, .1, "confirm")
        for source in SOURCES:
            np.testing.assert_array_equal(original[source]["values"], filtered[source]["values"])
        episodes = confirmation.aggregate_episodes(rows, expected, self.config)
        result = confirmation.known_delay_description(rows, episodes, expected, self.config)
        oracle = next(r for r in result if r["scope"] == "Huawei" and r["case"] == "PRIMARY_POOLED"
                      and r["controller"] == confirmation.ORACLE)
        self.assertEqual((oracle["status"], oracle["N_app_case_cells"], oracle["distinct_apps"]), ("OK", 8, 2))
        self.assertAlmostEqual(oracle["risk_mean"], .02)
        self.assertAlmostEqual(oracle["paid_ratio_to_nominal"], .2)
        self.assertEqual(oracle["compliance"], 1.)
        self.assertEqual(len(result), 75)
        changed = copy.deepcopy(rows)
        next(r for r in changed if r["role"] == "oracle_locked" and r["phase"] == "test")["candidate"] = 1
        bad = confirmation.known_delay_description(changed, confirmation.aggregate_episodes(changed, expected, self.config), expected, self.config)
        self.assertTrue(any(r["status"] == "INCOMPLETE" and r["scope"] == SOURCES[0] for r in bad))
        missing = [r for r in rows if not (r["role"] == "oracle_candidate" and r["candidate"] == 29)]
        self.assertEqual(confirmation.oracle_audit(missing, expected, self.config)[0], set())

    def test_confirmation_keeps_all_thirteen_cases_and_adverse_flash(self):
        rows, expected = self.confirmation_fixture()
        for row in rows:
            if row["phase"] == "test" and row["scenario"] == "flash5x60":
                row.update(violations=200, contract=int(row["replicate"] != 71))
        episodes = confirmation.aggregate_episodes(rows, expected, self.config)
        described = confirmation.all_case_description(episodes, expected)
        self.assertEqual(len(described), 5 * 13 * 9 * 3)
        self.assertEqual({r["case"] for r in described}, set(confirmation.CASES))
        self.assertEqual({r["controller"] for r in described}, set(confirmation.CONTROLLERS))
        flash = next(r for r in described if r["scope"] == SOURCES[0] and r["case"] == "flash5x60"
                     and r["controller"] == "BACC-lite" and r["contract_scope"] == "outside_contract")
        self.assertEqual((flash["status"], flash["N_app_case_cells"], flash["distinct_apps"]), ("OK", 1, 1))
        self.assertAlmostEqual(flash["risk_mean"], .2)
        self.assertEqual(flash["compliance"], 0.)

    def test_confirmation_five_eta_descriptions_never_reselect_or_overwrite_exploration(self):
        rows, expected = self.confirmation_fixture()
        with mock.patch.object(analysis, "select_eta", side_effect=AssertionError("Confirmation cannot reselect")):
            risk, described = confirmation.confirmation_descriptions(rows, expected, self.config)
        self.assertEqual(len(risk), 40)
        sensitivity = [r for r in described if r["section"] == "eta_sensitivity"]
        self.assertEqual({r["variant"] for r in sensitivity}, {f"eta_{eta}" for eta in self.config["search"]["wrapper_eta"]})
        flags = [r for r in sensitivity if r["metric"] == "frozen_primary_eta"]
        self.assertEqual({r["variant"] for r in flags if r["value"] == 1}, {"eta_0.1"})
        self.assertFalse(any(r["metric"] in ("selected", "selection_reason", "selection_eligible") for r in sensitivity))
        adverse = next(r for r in sensitivity if r["variant"] == "eta_0.5" and r["metric"] == "p3_compliance" and r["scope"] == "overall")
        self.assertEqual(adverse["value"], 0.)
        self.assertNotEqual(analysis.endpoint_destination("confirm"), analysis.endpoint_destination("explore"))
        self.assertEqual(analysis.endpoint_destination("confirm").name, "v2-lite-confirm-endpoints.csv")

    def test_confirmation_case_sensitivity_covers_five_eta_all_cases_with_fixed_weights(self):
        rows, expected = self.confirmation_fixture((1, 3, 1))
        for row in rows:
            if row["role"] == "locked" and row["phase"] == "test" and row["scenario"] == "flash5x60":
                if row["wrapped"] and row["eta"] == .2:
                    row["violations"] = 200 if row["source"] == SOURCES[0] else 800
            if row["role"] == "locked" and row["phase"] == "test" and row["scenario"] == "ramp2":
                row["contract"] = 0
        before = analysis.panels(rows, expected, .1, "confirm")
        with mock.patch.object(analysis, "select_eta", side_effect=AssertionError("No confirmation reselection")):
            risk, described = confirmation.confirmation_descriptions(rows, expected, self.config)
        sensitivity = [r for r in described if r["section"] == "eta_case_sensitivity"]
        self.assertEqual(len(sensitivity), 5 * 13 * 4 * 2 * 6 * 3)
        self.assertEqual({(r["eta"], r["case"]) for r in sensitivity},
                         {(eta, case) for eta in self.config["search"]["wrapper_eta"] for case in confirmation.CASES})
        def point(scope, case, eta=.2, population="all_selected"):
            return next(r for r in sensitivity if r["scope"] == scope and r["case"] == case and r["eta"] == eta
                        and r["controller"] == "HPA_wrapped" and r["contract_scope"] == population)
        self.assertAlmostEqual(point("Huawei", "flash5x60")["risk_mean"], .5)
        self.assertEqual(point("Huawei", "flash5x60")["N_app_case_cells"], 4)
        self.assertAlmostEqual(point("overall", "flash5x60")["risk_mean"], .65)
        self.assertEqual(point("Huawei", "ramp2", population="contract")["status"], "EMPTY_SUPPORT")
        self.assertEqual(point("Huawei", "ramp2", population="outside_contract")["status"], "OK")
        after = analysis.panels(rows, expected, .1, "confirm")
        for source in SOURCES:
            np.testing.assert_array_equal(before[source]["values"], after[source]["values"])
        main_flash = next(r for r in described if r["section"] == "all_13_cases" and r["scope"] == "Huawei"
                          and r["case"] == "flash5x60" and r["controller"] == "HPA_wrapped" and r["contract_scope"] == "all_selected")
        self.assertEqual(main_flash["eta"], .1)
        self.assertAlmostEqual(main_flash["risk_mean"], .05)


if __name__ == "__main__":
    unittest.main()
