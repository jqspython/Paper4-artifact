"""Synthetic source/ID frames and tiny local archives only; no confirmation IO."""

import copy
import csv
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

from scripts import prepare_v2_lite_confirmation as prep
import v2_lite


class ConfirmationDataProperties(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(dir=prep.ROOT / "work", prefix="confirm-data-property-")
        self.root = Path(self.folder.name)
        self.addCleanup(self.folder.cleanup)
        self.config = copy.deepcopy(prep.load_config())
        self.config["io"].update(exploration_data="old", confirmation_data="confirm")
        self.pool = {s: [f"toy-{s}-{i:05}" for i in range(n)] for s, n in prep.POOL_COUNTS.items()}
        self.old = {s: ids[:50] for s, ids in self.pool.items()}
        self.texts = {}
        (self.root / "old").mkdir()
        for name, groups in (("eligible-unit-pool.csv", self.pool), ("selected-unit-ids.csv", self.old)):
            text = io.StringIO()
            writer = csv.writer(text, lineterminator="\n")
            writer.writerow(("source", "unit_id"))
            writer.writerows((s, a) for s in prep.SOURCES for a in groups[s])
            self.texts[name] = text.getvalue()
            (self.root / "old" / name).write_text(text.getvalue(), encoding="utf-8")
        self.patches = [mock.patch.object(prep, "ROOT", self.root),
                        mock.patch.object(v2_lite, "ROOT", self.root)]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def fake_git(self, *args):
        return prep.DESIGN_SHA if args[0] == "rev-parse" else self.texts[args[1].rsplit("/", 1)[-1]]

    def make_manifest(self):
        with mock.patch.object(prep, "git", side_effect=self.fake_git), \
             mock.patch.object(prep, "require_stage_gate", return_value="synthetic-confirm-ref"):
            return prep.prepare("identities", self.config)

    def test_gate_precedes_frames_manifest_arrays_archives_and_outputs(self):
        for stage in ("identities", "demands"):
            with mock.patch.object(prep, "require_stage_gate", side_effect=RuntimeError("not frozen")) as gate, \
                 mock.patch.object(prep, "frozen_frame", side_effect=AssertionError("identity frame read")), \
                 mock.patch.object(prep, "validate_manifest", side_effect=AssertionError("manifest read")), \
                 mock.patch.object(prep.np, "load", side_effect=AssertionError("NPZ read")), \
                 mock.patch.object(prep, "raw_inventory", side_effect=AssertionError("archive touched")):
                with self.assertRaisesRegex(RuntimeError, "not frozen"):
                    prep.prepare(stage, self.config)
                gate.assert_called_once_with(self.config, "confirm", experiment=True)
            self.assertFalse((self.root / "confirm").exists())

    def test_identities_uses_only_frozen_frames_and_never_demand(self):
        with mock.patch.object(prep, "git", side_effect=self.fake_git), \
             mock.patch.object(prep, "require_stage_gate", return_value="synthetic-confirm-ref"), \
             mock.patch.object(prep.np, "load", side_effect=AssertionError("NPZ read")), \
             mock.patch.object(prep, "raw_inventory", side_effect=AssertionError("archive metadata read")), \
             mock.patch.object(prep, "huawei_selected", side_effect=AssertionError("Huawei body")), \
             mock.patch.object(prep, "azure_selected", side_effect=AssertionError("Azure body")):
            selected = prep.prepare("identities", self.config)
            self.assertEqual([len(selected[s]) for s in prep.SOURCES], [57, 80, 80])
            for source in prep.SOURCES:
                expected = sorted(set(self.pool[source]) - set(self.old[source]), key=lambda a: (
                    prep.hashlib.sha256(("20261020-v2-confirm:" + a).encode()).digest(), a))[:80]
                self.assertEqual(list(selected[source]), expected)
                self.assertFalse(set(selected[source]) & set(self.old[source]))
            path = self.root / "confirm/selected-unit-ids.csv"
            before = path.read_bytes()
            self.assertEqual(prep.prepare("identities", self.config), selected)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual([p.name for p in path.parent.iterdir()], [path.name])

    def test_frozen_population_duplicates_and_config_drift_are_rejected(self):
        name = "eligible-unit-pool.csv"
        self.texts[name] += f"{prep.SOURCES[0]},{self.pool[prep.SOURCES[0]][0]}\n"
        with mock.patch.object(prep, "git", side_effect=self.fake_git):
            with self.assertRaisesRegex(ValueError, "counts/duplicates"):
                prep.frozen_frame(self.config)
        changed = copy.deepcopy(self.config)
        changed["selection"]["hash_prefix"] = "different:"
        with self.assertRaisesRegex(ValueError, "hash/cap"):
            prep.selected_ids(self.pool, self.old, changed)

    def test_bad_manifest_stops_before_any_raw_access(self):
        self.make_manifest()
        path = self.root / "confirm/selected-unit-ids.csv"
        text = path.read_text("utf-8")
        path.write_text(text.replace("toy-", "changed-", 1), encoding="utf-8")
        with mock.patch.object(prep, "require_stage_gate", return_value="synthetic-confirm-ref"), \
             mock.patch.object(prep, "raw_inventory", side_effect=AssertionError("raw metadata read")), \
             mock.patch.object(prep, "huawei_selected", side_effect=AssertionError("Huawei body")), \
             mock.patch.object(prep, "azure_selected", side_effect=AssertionError("Azure body")):
            with self.assertRaisesRegex(ValueError, "Prepared identities differ"):
                prep.prepare("demands", self.config)

    def test_missing_raw_never_downloads_or_substitutes(self):
        self.make_manifest()
        with mock.patch.object(prep, "require_stage_gate", return_value="synthetic-confirm-ref"), \
             mock.patch.object(prep, "git", side_effect=self.fake_git), \
             mock.patch.object(prep.legacy, "RAW", self.root / "missing-raw"), \
             mock.patch.object(prep.legacy, "download", side_effect=AssertionError("download")), \
             mock.patch.object(prep, "huawei_selected", side_effect=AssertionError("raw body")):
            with self.assertRaisesRegex(ValueError, "NOT_PREPARED"):
                prep.prepare("demands", self.config)

    def test_all_identities_validate_before_extractors_and_saves(self):
        expected = self.make_manifest()
        events = []
        shared = prep.validate_manifest
        def checked(*args, **kwargs):
            events.append("validated")
            return shared(*args, **kwargs)
        def huawei(source, ids):
            self.assertIn("validated", events)
            self.assertEqual(tuple(ids), tuple(expected[source]))
            events.append(source)
            return np.ones((len(ids), 1)), np.ones((len(ids), 1), dtype=bool)
        def azure(ids):
            return huawei("azure2019", ids)
        with mock.patch.object(prep, "require_stage_gate", return_value="synthetic-confirm-ref"), \
             mock.patch.object(prep, "git", side_effect=self.fake_git), \
             mock.patch.object(prep, "validate_manifest", side_effect=checked), \
             mock.patch.object(prep, "raw_inventory", side_effect=lambda: events.append("inventory")), \
             mock.patch.object(prep, "huawei_selected", side_effect=huawei), \
             mock.patch.object(prep, "azure_selected", side_effect=azure), \
             mock.patch.object(prep, "save_source", side_effect=lambda *args: events.append("save")) as save:
            prep.prepare("demands", self.config)
        self.assertEqual(events[:2], ["validated", "inventory"])
        self.assertEqual(save.call_count, 3)

    def test_selected_huawei_columns_clock_and_missing_mask(self):
        source = prep.SOURCES[0]
        path = self.root / "fixture.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for day, value in ((28, 3), (29, 7)):
                text = io.StringIO()
                writer = csv.writer(text, lineterminator="\n")
                writer.writerow(("day", "time", "chosen", "ignored"))
                for minute in range(1440):
                    cell = "" if minute == 5 else "NaN" if minute == 6 else value
                    writer.writerow((day, day * 86400 + minute * 60, cell, "not-a-number"))
                archive.writestr(f"requests_minute/day_{day}.csv", text.getvalue())
        specification = (path.name, "unused", path.stat().st_size, 28, 29)
        with mock.patch.object(prep.legacy, "RAW", self.root), mock.patch.dict(prep.legacy.SOURCES, {source: specification}):
            demand, observed = prep.huawei_selected(source, ["chosen"])
            self.assertEqual(demand.shape, (1, 2880))
            self.assertEqual((demand[0, 0], demand[0, 1440]), (3, 7))
            self.assertTrue(observed[0, 0])
            self.assertFalse(observed[0, 5:7].any())
            np.testing.assert_array_equal(demand[0, 5:7], 0.)
            with self.assertRaisesRegex(ValueError, "identity absent"):
                prep.huawei_selected(source, ["not-present"])

    def test_selected_azure_uses_legacy_app_sum_and_absent_day_mask(self):
        path = self.root / "fixture.tar.xz"
        with tarfile.open(path, "w:xz") as archive:
            for day in (1, 2):
                text = io.StringIO()
                writer = csv.writer(text, lineterminator="\n")
                writer.writerow(["HashApp", *map(str, range(1, 1441))])
                if day == 1:
                    writer.writerow(["chosen", *([1] * 1440)])
                    writer.writerow(["chosen", *([3] * 1440)])
                writer.writerow(["ignored", *(["not-a-number"] * 1440)])
                payload = text.getvalue().encode()
                member = tarfile.TarInfo(f"invocations_per_function_md.anon.d{day:02}.csv")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
        specification = (path.name, "unused", path.stat().st_size, 1, 2)
        with mock.patch.object(prep.legacy, "RAW", self.root), mock.patch.dict(prep.legacy.SOURCES, {"azure2019": specification}):
            demand, observed = prep.azure_selected(["chosen"])
        np.testing.assert_array_equal(demand[0, :1440], 4.)
        self.assertTrue(observed[0, :1440].all())
        np.testing.assert_array_equal(demand[0, 1440:], 0.)
        self.assertFalse(observed[0, 1440:].any())

    def test_saved_npz_is_lossless_complete_and_never_overwrites(self):
        output = self.root / "saved"
        output.mkdir()
        source = prep.SOURCES[0]
        demand = np.full((1, 33 * 1440), 16777217., dtype=np.float64)
        observed = np.ones(demand.shape, dtype=bool)
        identity = "toy-" + "a" * 80
        prep.save_source(output, source, [identity], demand, observed)
        with np.load(output / f"{source}.npz", allow_pickle=False) as data:
            self.assertEqual(set(data.files), {"demand", "observed", "unit_id", "fit_end", "cal_end", "val_end", "first_day"})
            np.testing.assert_array_equal(data["demand"], demand)
            np.testing.assert_array_equal(data["observed"], observed)
            self.assertEqual(data["unit_id"].tolist(), [identity])
            self.assertEqual(tuple(int(data[k]) for k in ("fit_end", "cal_end", "val_end", "first_day")), (4320, 7200, 11520, 28))
        with self.assertRaisesRegex(ValueError, "not overwritten"):
            prep.save_source(output, source, [identity], demand, observed)
        self.assertFalse(list(output.glob("*.tmp")))

    def test_output_cannot_escape_project_or_replace_exploration(self):
        for destination in ("../outside", "old"):
            changed = copy.deepcopy(self.config)
            changed["io"]["confirmation_data"] = destination
            with self.assertRaisesRegex(ValueError, "separate project-local"):
                prep.output_directory(changed)


if __name__ == "__main__":
    unittest.main()
